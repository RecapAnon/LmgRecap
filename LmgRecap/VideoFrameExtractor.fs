module VideoFrameExtractor

open System
open System.Diagnostics
open System.IO

let getVideoDuration videoFilePath =
    let parameters =
        $"-v error -show_entries format=duration -of \"csv=p=0\" \"{videoFilePath}\""

    use proc = new Process()
    proc.StartInfo.FileName <- "ffprobe"
    proc.StartInfo.Arguments <- parameters
    proc.StartInfo.RedirectStandardOutput <- true
    proc.StartInfo.UseShellExecute <- false
    proc.StartInfo.CreateNoWindow <- true

    proc.Start() |> ignore
    let output = proc.StandardOutput.ReadToEnd()
    proc.WaitForExit()

    if proc.ExitCode <> 0 then
        raise (Exception("Failed to get video duration"))

    Double.Parse(output)

let captureFrame videoFilePath time outputPath =
    let t = sprintf "%0.3f" time
    let parameters = $"-ss {t} -i \"{videoFilePath}\" -vframes 1 \"{outputPath}\""

    use proc = new Process()
    proc.StartInfo.FileName <- "ffmpeg"
    proc.StartInfo.Arguments <- parameters
    proc.StartInfo.RedirectStandardOutput <- true
    proc.StartInfo.UseShellExecute <- false
    proc.StartInfo.CreateNoWindow <- true

    proc.Start() |> ignore
    proc.WaitForExit()

    if proc.ExitCode <> 0 then
        raise (Exception("Failed to capture frame"))

let getMiddleFrameBytes videoFilePath =
    let duration = getVideoDuration videoFilePath
    let middleTime = duration / 2.0

    let outputPath = Path.GetTempFileName()
    File.Delete(outputPath)
    let outputPath = Path.ChangeExtension(outputPath, "jpg")

    captureFrame videoFilePath middleTime outputPath

    let imageBytes = File.ReadAllBytes(outputPath)

    File.Delete(outputPath)

    imageBytes
