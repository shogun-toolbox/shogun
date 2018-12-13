import sys
import json
try:
    import urllib.request as urllib
except:
    import urllib2 as urllib
import os
import shutil
import zipfile


target_branch = sys.argv[1]
artifact_name = sys.argv[2]
url = 'https://dev.azure.com/shogunml/shogun/_apis/build/builds?api-version=4.1&resultFilter=succeeded&branchName=refs/heads/%s' % (target_branch)
response = urllib.urlopen(url)
content = json.loads(response.read())
if content['count'] > 0:
    latest_build_id = content['value'][0]['id']
    artifact_url = 'https://dev.azure.com/shogunml/shogun/_apis/build/builds/%d/artifacts?artifactName=%s&api-version=4.1' % (latest_build_id, artifact_name)
    response = urllib.urlopen(artifact_url)
    content = json.loads(response.read())
    zip_url = content['resource']['downloadUrl']
    file_name = '%s.zip' % (artifact_name)

    with open(file_name, 'wb') as out_file:
        data = urllib.urlopen(zip_url)
        shutil.copyfileobj(data, out_file)

    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())
else:
    exit(1)