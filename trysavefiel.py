path = 'viewData.ipynb'

with open(path, 'rb') as f:
    text = f.read()
text = text.encode('utf8')
print(text)
# new_file_path = 'new/path'
# with open(new_file_path, 'wb') as f:
#     f.write(text)