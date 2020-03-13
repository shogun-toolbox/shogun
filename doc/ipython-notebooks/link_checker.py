import json
import re
import subprocess
import sys


#requirement is python >= 3.5

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
    process = subprocess.Popen(['curl','-s', '-o','/dev/null','-I','-w','%{http_code}',link_address],stdout=subprocess.PIPE, stderr=subprocess.PIPE,universal_newlines=True)
    stdout, stderr = process.communicate()
    return (int(stdout))





if __name__ == '__main__':

    # enter name of file
    file_name = sys.argv[1]


    links = link_finder(file_name)

    
    number_of_links = len(links)
    
    for i in range(number_of_links):
        return_code = link_checker(links[i])
        if(return_code != 200):
            print(links[i])


    

    

   

    
          
    