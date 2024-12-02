open System
open System.IO
open System.Text
open System.Text.RegularExpressions

let findFirstAlpha (text: string) : string option =
    let m = Regex.Match(text, "[a-zA-Z]")
    if m.Success then Some(m.Value) else None

let loadAndSortSections () =
    let mutable sections = []

    for path in Directory.EnumerateFiles(".", "*.md", SearchOption.AllDirectories) do
        let content = File.ReadAllText(path)
        let parts = content.Split([| "\r\n\r\n" |], StringSplitOptions.RemoveEmptyEntries)

        for part in parts do
            match findFirstAlpha (part) with
            | Some(firstAlpha) -> sections <- (firstAlpha, part.Trim()) :: sections
            | None -> ()

    let sortedSections =
        List.sortBy (fun ((alpha: string), (content: string)) -> content.ToLower()) sections

    use f = new StreamWriter(File.OpenWrite("glossary.md"), Encoding.UTF8)
    sortedSections |> List.iter (fun (_, section) -> f.WriteLine(section + "\r\n"))

loadAndSortSections()
