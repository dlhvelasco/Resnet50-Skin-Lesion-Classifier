import subprocess

print("Printing final results...\n")
finalresult = ['python',r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/get_results.py',r'C:/Users/Dwight/Music/AP187-Imaging/Dermatologist-AI/predictions.csv','0.5']
finalresultcomplete = subprocess.run(finalresult, shell=True)