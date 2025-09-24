Usage:

## In the Browser:

- (1) Go to Firebase Console and create a new project (or use an existing one)
- (2) Create a Wep App in the Project Setting (if you haven't already)
    - If you setting up a Project, copy the firebaseConfig into index.html (in the docs folder)
    - If you have already setup a project before, you find the firebaseConfig here:
        Click in the top-left corner on the gear icon (Project Settings) -> scroll down a little bit
- (3) Create a Firestore Database
  - Click on Build (left sidebar) -> Firestore Database -> Create Database
  - Choose a location and start in production Mode
  - Copy the rules from `testing_zone/firestore.rules` into the rules tab and publish them
- (4) Get the Service Account Key
  - Next to Project Overview, click on the gear icon (Project Settings)
  - On the top, click on the tab "Service Accounts"
  - Generate a new private key and store the downloaded file in the researcher_hub folder of this project
    under the name `firebase_credentials.json`

## In the Project (Terminal)
- (5) CD into the testing_zone in this project and run the following commands: 

```shell
cd testing_zone
```

```shell
npm install firebase
```

```shell
npm install -g firebase-tools
```

```shell
firebase login
```

```shell
firebase init
```

> - Choose only Firestore (not Hosting, Functions, etc)
> - Use an existing project (the one you just created)
> - for the database you can choose nam5 and us-central (or whatever is close to you)
> - ? What file should be used for Firestore Rules? > firestore.rules -> enter
> - If asked to overwrite, choose no
> - ? What file should be used for Firestore indexes? > firestore.indexes.json -> enter

- (6) Now you can deploy the firestore rules:
```shell
firebase deploy
```

- (7) Cd into the researcher_hub folder and install the requirements:

```shell
cd ../researcher_hub
```

First, make sure your virtual environment is activated (see above)

If you don't have a virtual environment yet, create one and activate it:

```shell
python -m venv .venv
```

```shell
source .venv/bin/activate
```

Then install the requirements:

```shell
pip install -r requirements.txt
```

- (8) Now you can run the researcher hub:

```shell
python autora_workflow.py
```

- (9) Wait until the conditions are uploaded. (You can see this in your Browser in the Firestore Database)

- (10) As soon as the terminal starts spamming a red string you can open the `docs/index.html` in a browser
and should see the experiment.

- (11) You can test the experiment twice (because there are two conditions). For now, the experiment will end 
"abruptly" after a few trials. You notice when there are no new stimuli appearing.

## Set up as github pages to host the experiment:

- (12) Push everything to github (make sure firebase_credentials.json is in .gitignore so you don't expose your credentials)
- (13) Go to the repository on github.com
- (14) Go to Settings -> Pages (on the left sidebar)
- (15) Under "Build and deployment" select "Deploy from a branch"
- (16) Under "Branch" select "main" and "/docs" folder
- (17) Save
- (18) After a few minutes, your experiment should be available under https://<your-github-username>.github.io/<your-repo-name>/
- (19) Success: You can run the autora_workflow.py in this projcet to upload new conditions to firestore and do the experiment 
on the gitHub pages link