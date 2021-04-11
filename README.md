# basic-net-js

You only need to do the below if you want to actually make changes to the code that I can download, if you just want to look at the code, click the 'Code' tab.

## Setup

1. Download and install the LTS version of node.js from here: https://nodejs.org/en/ (if it asks if it should add stuff to your path click yes or tick the box)
2. Download and install VScode from here: https://code.visualstudio.com/

## Clone Git Repo (get the code from github with version control)

A bit about git in general: https://www.youtube.com/watch?v=hwP7WQkmECE

I suggest you use Sourcetree as your git GUI, I use it and find it makes connecting to and managing git repos easier and less command line driven.

### Setup 

1. Download and install Sourcetree from here: https://www.sourcetreeapp.com/ 
2. Open Sourcetree
3. You may be asked to login to an account or setup some basic settings such as name and email, git needs those.
4. File -> New
5. Set source path to: https://github.com/CassMidnight/basic-net-js
6. Click the 'browse' button next to set desinaion path
7. Create a new folder in an appropriate direcory and select it
8. Click the clone button

### Push changes

You can use any git tool you want but here is the how to for Sourcetree.

1. Open sourceree
2. Click the plus button next to any of the files you have changed, this will move them into the "staged" area 
3. Click the commit button to confirm your changes, you will need to provide a short commit message saying what you changed
4. Click the pull button, this will download any changes I have made
5. If the changes I have made could be merged without causing any conflicts then any changes i have made will show up
6. If the changes I have made could NOT be merged without causing conflict then source tree will ask you to resolve the conflicts in its merge tool 
7. Resolve the conflicts and click commit
8. Click the push button to push all your changes to the server

## Run

1. Open the folder with the project in VScode 
2. Run the below command in the VScode terminal

`node .\iris.js`  

If the output in the terminal is too long for the scrollback use the below to redirect the output to a file called output.txt in the same directory.

`node .\iris.js > output.txt`
