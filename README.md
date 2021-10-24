# show-tell

-------
BEFORE YOU START:
-------

>open cmd/anaconda terminal and cd to the folder with requirements.txt
>use the command: pip3 install -r requirements.txt

-------
HOW TO INSTALL MYSQL AND IMPORT DATA:
-------

>Install MySQL from this link:
https://dev.mysql.com/downloads/file/?id=506568

>install using the default settings, and when asked for a password, use 'root'

>launch MySQL Workbench > open local instance > create a new schema and name it as 'flaskapp'

>import data by going to Administration > data import/restore > select the folder Dump20211024 > start import

>you should now have the database 'flaskapp', which contain the table 'dog' and 'cat', which contain info like Breed, Descriptions, AverageLifeSpan, Characteristic, etc..

------
TO START THE SERVER:
------

>open cmd/anaconda terminal and cd to the folder with the app.py file

>use the command: python app.py

>copy the address (i.e. 127.0.0.1:3000/ ) and paste it in your browser i.e chrome, firefox, etc

------
KNOWN ERRORS/BUGS
------
>if you are getting a yaml error asking for a Loader() function, make sure that your PyYaml version == 5.4.1 by typing the command: pip install PyYAML==5.4.1

>TTS only saves the first result and not subsequent ones. need to refresh the page to get a new TTS audio file.
