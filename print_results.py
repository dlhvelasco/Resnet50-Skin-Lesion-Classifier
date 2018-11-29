import subprocess

print("Printing final results...\n")
finalresult = ['python',r'get_results.py',r'predictions.csv','0.5']
finalresultcomplete = subprocess.run(finalresult, shell=True)