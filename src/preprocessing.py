import json
import os
import xmltodict
import re
import time
import pandas as pd
from jsonpath import jsonpath
import numpy as np

# Initialize a dictionary to store the data
data = {
    "Questions": [],
    "Answers": [],
    "Focus": []
}

# Function to process an XML file
def processXmlFile(completePath):
    # Open the XML file
    with open(completePath, encoding = 'utf-8') as f:
        # Read the contents of the file
        xmlstring = f.read()

        try:
            # Parse the XML string into a dictionary using xmltodict library
            dataDict = xmltodict.parse(xmlstring, xml_attribs=False)
            
            # Extract the QAPair and Focus information from the dictionary
            listOfQA = json.loads(json.dumps(jsonpath(dataDict, '$..' + "QAPair")[0]))
            focus = json.loads(json.dumps(jsonpath(dataDict, '$..' + "Focus")[0]))
        except Exception as e:
            # Handle exceptions, such as empty QAPair or Focus
            return

        # Check if there is only a single QA pair, and convert it to a list if needed
        if isinstance(listOfQA, dict):
            listOfQA = [listOfQA]
        
        # Process each QA pair
        for qaPair in listOfQA:
            try:
                # Clean up the answer text
                x = re.sub(' +', ' ', qaPair['Answer'])
                x = re.sub('Key Points', "", x)
                x = x.replace("\n", "").replace("-", "")
                
                # Append the processed data to the data dictionary
                data['Answers'].append(x)
                data['Questions'].append(qaPair['Question'])
                data['Focus'].append(focus)
            except:
                # Handle any exceptions that occur during processing
                return
            
# List of folders with empty answers
foldersWithEmptyAnswers = [
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
]

# Base path for the folders
BASE_PATH = "./MedQuAD-master"

# Iterate over the folders in the base path
for folder in os.listdir(BASE_PATH):
    # Check if the folder is in the list of folders with empty answers
    if folder in foldersWithEmptyAnswers:
        # If the folder is in the list, skip it and continue with the next folder
        continue
    else:
        # If the folder is not in the list, process it
        print("Processing folder:", folder)
        start = time.time()

        # Iterate over the XML files in the current folder
        for xmlFileName in os.listdir(os.path.join(BASE_PATH, folder)):
            completePath = os.path.join(BASE_PATH, folder, xmlFileName)
            
            # Process the XML file
            processXmlFile(completePath)

        print("Took", time.time() - start)

# After creating the DataFrame
df = pd.DataFrame(data)
df.to_csv('processed_medquad.csv', index=False)