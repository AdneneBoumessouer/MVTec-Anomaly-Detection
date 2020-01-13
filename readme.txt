TO DO:

- see if loading pretrained model needs to recompile??
- see how to save pretrained model after training for a second time (update history.csv)


Observations:
Old model seems to work better...possible causes: change in architecture (MaxPool2D) and loss function (1 - MSSIM)




Retrieve local requirements:
pip freeze > requirements.txt
pip install -r requirements.txt 



Github:
url = git@github.com:AdneneBoumessouer/AutPr.git