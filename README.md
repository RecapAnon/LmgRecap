4chan Thread Recap Generator
============================

Overview
--------

This project generates recap summaries of 4chan threads, specifically for threads about running large language models on local hardware. The recap generator uses a combination of natural language processing (NLP) and computer vision techniques to analyze the thread and identify key posts, images, and discussions.

Features
--------

* Analyzes 4chan threads and identifies key posts, images, and discussions
* Uses NLP techniques to understand the content of the thread
* Employs computer vision to analyze images and identify relevant information
* Generates a recap summary of the thread, including key points and images
* Supports various options for customizing the recap generation process, including minimum rating, maximum replies, and logging level

Requirements
------------

* .NET Core 3.1 or later
* F# 4.7 or later
* Python 3.9 or later (for Python dependencies)
* OpenAI API key (for NLP tasks)
* 4chan API credentials (for accessing thread data)
* Firefox browser (for Selenium web scraping)

Installation
------------

1. Clone the repository using Git: `git clone https://github.com/your-username/4chan-thread-recap-generator.git`
2. Install the required .NET Core and F# dependencies using NuGet: `dotnet restore`
3. Install the required Python dependencies using pip: `pip install -r requirements.txt`
4. Create a `appsettings.json` file in the project root with your OpenAI API key and 4chan API credentials

Usage
-----

The recap generator can be run using the following command:
```
dotnet run --threadNumber <thread_number> --filename <filename> --postNumber <post_number> --rating <rating>
```
Replace `<thread_number>` with the ID of the 4chan thread you want to generate a recap for, `<filename>` with the path to the Markdown file containing the thread data, `<post_number>` with the ID of the post you want to override the rating for, and `<rating>` with the new rating.

Available commands:

* `recap`: Generates a recap summary of the specified thread
* `load-memories`: Loads memories from a Markdown file
* `print-recap`: Prints the recap summary to the console
* `override-rating`: Overrides the rating of a post in the recap summary
* `override-post-rating`: Overrides the rating of a post in the recap summary
* `drop-summary`: Drops the summary of a post in the recap summary
* `thread-summary`: Generates a summary of the thread

Options
-------

The recap generator supports the following options:

* `--MinimumRating`: Sets the minimum rating required for a post to be included in the recap summary
* `--MinimumChainRating`: Sets the minimum rating required for a chain of posts to be included in the recap summary
* `--MaxReplies`: Sets the maximum number of replies to include in the recap summary
* `--MaxLength`: Sets the maximum length of the recap summary
* `--Logging:LogLevel:Default`: Sets the logging level for the application
* `--RateMultiple`: Enables or disables the rating of multiple posts in the recap summary
* `--RateChain`: Enables or disables the rating of chains of posts in the recap summary
* `--Describe`: Enables or disables the generation of a description for the recap summary

Contributing
------------

Contributions to this project are welcome. Please submit a pull request with your changes and a brief description of what you've added or fixed.

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.