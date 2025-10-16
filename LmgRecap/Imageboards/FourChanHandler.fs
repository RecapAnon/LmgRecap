namespace LmgRecap.Imageboards

open System
open OpenQA.Selenium
open System.Text
open System.Text.RegularExpressions
open System.Web
open System.Text.Json
open OpenQA.Selenium.Firefox
open LmgRecap.Shared
open System.Linq

type FourChanHandler() =
    interface IImageboardHandler with
        member this.FetchThreadJson(driver, builder, maxId, filters: string array) =
            driver
                .Navigate()
                .GoToUrl($"https://a.4cdn.org/g/thread/{builder.ThreadId}.json")

            driver.FindElement(By.TagName("html")).Text
            |> JsonSerializer.Deserialize<Thread>
            |> fun a -> a.posts[1..]
            |> Array.filter (fun p -> p.no > maxId)
            |> Array.map FourChanPost
            |> Array.map (fun i -> (this :> IImageboardHandler).MapPostToChainNode(filters, i))

        member this.MapPostToChainNode(filters: string array, post) =
            match post with
            | FourChanPost p ->
                { id = p.no
                  timestamp = p.time
                  now = p.now
                  filtered = filters.Any(fun filter -> Regex.IsMatch((defaultArg p.com "").ToLower(), filter))
                  links = [||]
                  replies = [||]
                  ratings = [||]
                  comment = ""
                  unsanitized = defaultArg p.com ""
                  context = None
                  rating = -1
                  reasoning = ""
                  filename =
                    match p.tim.IsSome && p.ext.IsSome with
                    | true -> Some(p.tim.Value.ToString() + p.ext.Value.ToString())
                    | _ -> None
                  caption = None
                  label = None
                  confidence = None }
            | EightChanPost _ -> failwith "MapPostToChainNode is not applicable for 4chan"
            | MegucaPost _ -> failwith "MapPostToChainNode is not applicable for 4chan"

        member this.BuildReferences(chainmap, node) =
            let quoteIndicator = (this :> IImageboardHandler).QuoteIndicator

            { node with
                links =
                    Regex.Matches(
                        node.unsanitized,
                        quoteIndicator + @"(\d{" + node.id.ToString().Length.ToString() + "})"
                    )
                    |> Seq.map (fun m -> m.Groups[1].ToString() |> Int64.Parse)
                    |> Seq.filter (fun id -> chainmap.ContainsKey id)
                    |> Array.ofSeq
                replies =
                    chainmap.Values
                    |> Seq.filter (fun p -> p.unsanitized.Contains(node.id.ToString()))
                    |> Seq.map (fun p -> p.id)
                    |> Seq.toArray }

        member this.Sanitize(node, fetchTitleFromUrlAsync) =
            let stripTags (input: string) =
                let sb = StringBuilder()
                let mutable inTag = false

                for c in input do
                    match c with
                    | '<' -> inTag <- true
                    | '>' -> inTag <- false
                    | _ when not inTag -> sb.Append(c) |> ignore
                    | _ -> ()

                sb.ToString()

            match node.comment = String.Empty with
            | true ->
                { node with
                    comment =
                        node.unsanitized.Replace("<br>", "\n")
                        |> stripTags
                        |> HttpUtility.HtmlDecode
                        |> fun c -> c.Replace("https://arxiv.org/pdf", "https://arxiv.org/abs").Trim()
                        |> fun c ->
                            (Regex.Matches(
                                c,
                                @"https:\/\/([\w\-_]+(?:(?:\.[\w\-_]+)+))([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?"
                             )
                             |> Seq.map (fun m ->
                                 m.Value, fetchTitleFromUrlAsync m.Value |> Result.defaultValue "Error fetching title")
                             |> Seq.fold
                                 (fun (newText: string) (url, title) -> newText.Replace(url, $"[{title}]({url})"))
                                 c) }
            | _ -> node

        member this.QuoteIndicator = "&gt;&gt;"

        member this.GetDownloadLink(driver: FirefoxDriver, url: string) : IWebElement =
            driver.FindElement(By.CssSelector($"a[href='{url}'][download]"))

        member this.GetImageUrl(filename: string) : string = $"https://i.4cdn.org/g/{filename}"

        member this.GetThreadUrl(threadId: string) : string =
            $"https://boards.4chan.org/g/thread/{threadId}"
