module FSUtils

open Microsoft.SemanticKernel
open System.CommandLine

type Command = System.CommandLine.Command
type Argument<'T> = System.CommandLine.Argument<'T>
type RootCommand = System.CommandLine.RootCommand

let addAlias alias (command: Command) =
    command.AddAlias alias
    command

let addCommand subCommand (command: RootCommand) =
    command.AddCommand subCommand
    command

let addOption option (command: Command) =
    command.AddOption option
    command

let addGlobalOption option (command: RootCommand) =
    command.AddGlobalOption option
    command

let addArgument argument (command: Command) =
    command.AddArgument argument
    command

let setHandler (handler: string -> unit) argument (command: Command) =
    command.SetHandler(handler, argument)
    command

let setHandler2 handler argument1 argument2 (command: Command) =
    command.SetHandler(handler, argument1, argument2)
    command

let setHandler3 handler argument1 argument2 argument3 (command: Command) =
    command.SetHandler(handler, argument1, argument2, argument3)
    command

let invoke (argv: string array) (rc: RootCommand) = rc.Invoke argv

let set i v (a: KernelArguments) =
    a[i] <- v
    a
