<# run_all.ps1 — start bridge -> launch VPX -> run training #>

[CmdletBinding()]
param(
  [string]$ServerFile = "vpx_bridge.py",
  [string]$TrainFile  = "train.py",
  [int]$StartupDelaySeconds = 4
)

$ErrorActionPreference = "Stop"

function Load-DotEnv {
  if (Test-Path ".\.env") {
    $lines = Get-Content .\.env | Where-Object { $_ -match '^\s*[^#].*=' }
    foreach ($line in $lines) {
      if ($line -match '^\s*([^=]+)=(.*)$') {
        $name = $Matches[1].Trim()
        $val  = $Matches[2].Trim()
        if ($val.StartsWith('"') -and $val.EndsWith('"')) {
          $val = $val.Substring(1, $val.Length-2)
        }
        Set-Variable -Name $name -Value $val -Scope Script
      }
    }
  } else {
    throw ".env not found. Run setup.ps1 first."
  }
}

function Require-File($path) {
  if (-not (Test-Path $path)) {
    throw "Required file not found: $path"
  }
}

function Wait-PortOpen([int]$Port, [int]$TimeoutSec = 20) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    try {
      $conn = Test-NetConnection -ComputerName "127.0.0.1" -Port $Port -WarningAction SilentlyContinue
      if ($conn.TcpTestSucceeded) { return $true }
    } catch {}
    Start-Sleep -Milliseconds 300
  }
  return $false
}

function Start-Bridge {
  Write-Host "Starting bridge: $ServerFile ..."
  $script:BridgeProc = Start-Process -FilePath $PYTHON_EXE -ArgumentList $ServerFile -NoNewWindow -PassThru
}

function Start-VPX {
  Write-Host "Launching Visual Pinball..."

  # sanity checks
  if (-not $VPX_TABLE -or -not (Test-Path $VPX_TABLE)) {
    throw "VPX table not found at: $VPX_TABLE"
  }
  if (-not (Test-Path $VPX_PATH)) {
    throw "VPinballX.exe not found at: $VPX_PATH"
  }

  # Use -play so VPX opens the table directly (and plays vs editor)
  $args = @("-play", "`"$VPX_TABLE`"")

  # Set working directory to the table's folder (helps with relative assets)
  $wd = Split-Path -Path $VPX_TABLE -Parent

  $script:VPXProc = Start-Process -FilePath $VPX_PATH `
                                  -ArgumentList $args `
                                  -WorkingDirectory $wd `
                                  -PassThru
}


function Start-Train {
  Write-Host "Starting training: $TrainFile ..."
  $script:TrainProc = Start-Process -FilePath $PYTHON_EXE -ArgumentList $TrainFile -NoNewWindow -PassThru
}

function Cleanup {
  Write-Host "`nShutting down..."
  foreach ($p in @($TrainProc,$VPXProc,$BridgeProc)) {
    if ($p -and !$p.HasExited) {
      try { $p.CloseMainWindow() | Out-Null } catch {}
      Start-Sleep -Milliseconds 300
      if (!$p.HasExited) { try { $p.Kill() } catch {} }
    }
  }
}

try {
  Load-DotEnv

  Require-File $PYTHON_EXE
  Require-File $ServerFile
  Require-File $TrainFile
  Require-File $VPX_PATH
  if ($VPX_TABLE) { Require-File $VPX_TABLE }

  Start-Bridge
  if (-not (Wait-PortOpen -Port $BRIDGE_PORT -TimeoutSec 30)) {
    throw "Bridge on port $BRIDGE_PORT did not come up. Check $ServerFile."
  }

  Start-Sleep -Seconds $StartupDelaySeconds

  Start-VPX

  Start-Sleep -Seconds $StartupDelaySeconds

  Start-Train

  $TrainProc.WaitForExit()

} catch {
  Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
} finally {
  Cleanup
  Write-Host "Done."
}
