Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$full = New-Object System.Drawing.Bitmap($b.Width, $b.Height)
$g = [System.Drawing.Graphics]::FromImage($full)
$g.CopyFromScreen($b.X, $b.Y, 0, 0, $full.Size)
$full.Save("C:\Users\immer\living-portraits\shot_full.png")
$crop = New-Object System.Drawing.Bitmap(448, 256)
$gc = [System.Drawing.Graphics]::FromImage($crop)
$srcRect = New-Object System.Drawing.Rectangle(0, 0, 448, 256)
$dstRect = New-Object System.Drawing.Rectangle(0, 0, 448, 256)
$gc.DrawImage($full, $dstRect, $srcRect, [System.Drawing.GraphicsUnit]::Pixel)
$crop.Save("C:\Users\immer\living-portraits\shot_panels.png")
$g.Dispose(); $gc.Dispose(); $full.Dispose(); $crop.Dispose()
Write-Output "SHOT_SAVED"
