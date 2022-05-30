import os

"""
                      ************        ^
                      * CAUTION! *       /!\
                      ************      <--->

Modifying and executing this file could not only destroy your current
project, it could also affect other files higher in the file hierarchy.
As it is right now, it will eraase (after asking the user to confirm it)
all files in results folder. For that reason, we encourage you to use it wisely
"""

print("Are you COMPLETELY sure you want to erase all files in results folder?")
print("Type 'Yes' to confirm, any other input will close this execution: ")

answer = input()

if answer == "Yes":
    print("Erasing all files (not folder) in the results folder recursively")
    rootdir = "../results/"
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            os.remove(root + "/" +  file)
    print("Execution succesful!\n------------------------------")

else:
    print("Ending without touching those files\n------------------------------")