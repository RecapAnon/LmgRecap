module Logging

open Microsoft.Extensions.Logging
open OpenTelemetry
open OpenTelemetry.Resources
open OpenTelemetry.Logs
open OpenTelemetry.Trace
open System
open System.Diagnostics
open System.IO
open System.Net.Http

let logger<'a> =
    LoggerFactory
        .Create(fun builder -> builder.SetMinimumLevel(LogLevel.Trace) |> ignore)
        .AddFile("requests.log", LogLevel.Trace, null, false, Nullable(), 100, "{Message:lj}{NewLine}")
        .CreateLogger()

let resourceBuilder = ResourceBuilder.CreateDefault().AddService("Scratch")

let enrichHttpRequest (activity: Activity) (req: HttpRequestMessage) =
    if req.Method = HttpMethod.Post then
        req.Content.LoadIntoBufferAsync().Wait()
        let ms = new MemoryStream()
        req.Content.CopyToAsync(ms).Wait()
        ms.Seek(0L, SeekOrigin.Begin) |> ignore
        use reader = new StreamReader(ms)
        let content = reader.ReadToEnd()

        if req.RequestUri.AbsolutePath.Contains("completions") then
            logger.LogInformation("Request: {@Request}", content)

        activity.SetTag("requestBody", content) |> ignore

let enrichHttpResponse (activity: Activity) (res: HttpResponseMessage) =
    res.Content.LoadIntoBufferAsync().Wait()
    let ms = new MemoryStream()
    res.Content.CopyToAsync(ms).Wait()
    ms.Seek(0L, SeekOrigin.Begin) |> ignore
    use reader = new StreamReader(ms)
    let content = reader.ReadToEnd()

    if res.RequestMessage.RequestUri.AbsolutePath.Contains("completions") then
        logger.LogInformation("Response: {@Response}", content)

    activity.SetTag("responseBody", content) |> ignore

let tracerProvider =
    Sdk
        .CreateTracerProviderBuilder()
        .SetResourceBuilder(resourceBuilder)
        .AddSource("Scratch")
        .AddSource("Microsoft.SemanticKernel*")
        .AddHttpClientInstrumentation(fun opts ->
            opts.EnrichWithHttpRequestMessage <- Action<Activity, HttpRequestMessage>(enrichHttpRequest)
            opts.EnrichWithHttpResponseMessage <- Action<Activity, HttpResponseMessage>(enrichHttpResponse))
        //.AddConsoleExporter()
        .Build()

let loggerFactory l =
    LoggerFactory.Create(fun builder ->
        builder
            .SetMinimumLevel(l)
            .AddOpenTelemetry(fun opts ->
                opts.SetResourceBuilder(resourceBuilder) |> ignore
                // opts.AddConsoleExporter() |> ignore
                opts.IncludeFormattedMessage <- true)
            .AddConsole()
        |> ignore)

let tracer = tracerProvider.GetTracer("Scratch")
