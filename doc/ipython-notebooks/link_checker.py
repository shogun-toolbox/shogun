import json
import re
import urllib.request, urllib.error
import sys
#requirement is python >= 3.5

def check_python():
    if (sys.version_info < (3,5)):
        raise Exception("Python >= 3.5 required")

def Find(cell_contents):
    text = cell_contents[0]
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return urls   

def link_finder(file_name):
    links = []
    f = open(file_name)
    s = f.read()
    input_file = json.loads(s)
    #convert given file to python dictionary fromat
    cells_count = len(input_file['cells'])
    for i in range(cells_count):
        #Assuming all links are in markdown cells
        if(input_file['cells'][i]['cell_type'] == 'markdown'):
            urls_in_current_cell = Find(input_file['cells'][i]['source'])
            links.extend(urls_in_current_cell)
    return links       

def link_checker(link_address):
    try:
        urllib.request.urlopen(link_address)
    except urllib.error.HTTPError as e1:
        # Return code error (e.g. 404, 501, ...)
        return e1.code
    except urllib.error.URLError as e2:
        # Not an HTTP-specific error (e.g. connection refused)
        return(e2.reason)
    else:
        return(200)
    

if __name__ == '__main__':
    check_python()
    # enter name of file
    file_name = sys.argv[1]
    links = link_finder(file_name)
    
    for link in links:
        return_code = link_checker(link)
        if(return_code != 200):
            print(link) 