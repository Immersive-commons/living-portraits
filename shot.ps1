# Screen-grab of the panels. MUST run in the console session (session 1) or it captures a
# black desktop -- that is why it exists as the `lp-shot` scheduled task rather than as
# something you can run over SSH.
#
# Saves next to this script ($PSScriptRoot) instead of a hardcoded path. It used to write to
# C:\Users\immer\living-portraits -- the pre-migration supercommons2 path -- so a copy of this
# script run anywhere but that box would have failed or written somewhere nobody looks.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$full = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
$g = [System.Drawing.Graphics]::FromImage($full)
$g.CopyFromScreen($b.X, $b.Y, 0, 0, $full.Size)
$full.Save((Join-Path $PSScriptRoot "shot_full.png"))
$crop = New-Object System.Drawing.Bitmap(448, 256)
$gc = [System.Drawing.Graphics]::FromImage($crop)
$srcRect = New-Object System.Drawing.Rectangle(0, 0, 448, 256)
$dstRect = New-Object System.Drawing.Rectangle(0, 0, 448, 256)
$gc.DrawImage($full, $dstRect, $srcRect, [System.Drawing.GraphicsUnit]::Pixel)
$crop.Save((Join-Path $PSScriptRoot "shot_panels.png"))
$g.Dispose(); $gc.Dispose(); $full.Dispose(); $crop.Dispose()
Write-Output "SHOT_SAVED"
