#!/usr/bin/python3
# print ("hello world")
# print ("김영빈")

# if True:
#     print ("나는")
#     print ("바보이다")

# else:
#     print ("그런데")
#     print ("바보와 천재는 종이 한 장 차이이다")

    #!/usr/bin/python3

import sys

try:
    # open file stream
    file - open(file_name, "w")

except IOError:
    print ("Thear was an error writing to", file_name)
    sts.exit()
print ("Enter '", file_finish,)
print (" When finished")
while file_text != file_finish:
    file_text = raw_input("Enter text: ")
    if file_text != file_finish:
        # close the file
            file.close
            break
    file.write(file_text)
    file.write("\n")
file.close()
file_name = input("Enter filename: ")
if len(file_name) == 0:
    print("Next time please enter something")
    sts.exit()

try:
    file = open(file_name, "r")

except IOError:
    print ("There was an error reading file")
    sys.exit()
file_text = file.read()
file.close()
print (file_text)