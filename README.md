# DNA Fiber Assay Workflow Automation App

#### This project is done as a part of my Bachelor's Thesis Project. 
DNA fiber assays are a single-molecule technique used to study DNA replication dynamics. These assays provide critical insights into replication fork progression, stalling, and origin activation. The process involves fluorescent labeling of DNA with red and green signals, representing different nucleotide analogs incorporated during replication.
Eventually, the DNA track lengths have to be manually calculated using tools like ImageJ. This application is created to automate that workflow, by simply sending the image to the app, changing some parameters to remove noise from the image, and clicking Analyze.
The statistical output of the ratios of the DNA track lengths is returned. This project comes with a user-friendly interface to work with, and the user can get the required data in seconds as compared to the hours required in manual tools.

#### Tech Stack used:
- Python
- OpenCV
- FastAPI
- React/Vite
- JavaScript

#### How to use it:
Start server:
- `cd server` and `pip install -r requirements.txt`
- `uvicorn main:app --reload --port 8000`

Start front-end:
- `cd client` and `npm i`
- `npm run dev`
