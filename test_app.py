from flask import Flask
import json
import io
import os

from flask.helpers import url_for
import pytest
from app import app

def test_home():
    client = app.test_client()
    url = '/'

    response = client.get(url)
    assert response.status_code == 200
    assert b"Loving Home for Pets" in response.data

def test_about():
    client = app.test_client()
    url = '/about.html'

    response = client.get(url)
    assert response.status_code == 200
    assert b"Data Architect" in response.data

def test_navBarNew():
    client = app.test_client()
    url = '/navbar.html'

    response = client.get(url)
    assert response.status_code == 200
    assert b"Opening Hour" in response.data

def test_footerNew():
    client = app.test_client()
    url = '/navbarfooter.html'

    response = client.get(url)
    assert response.status_code == 200
    assert b"461 Clementi Rd, Singapore 599491" in response.data

def test_dogs():
    client = app.test_client()
    url = '/dogs.html'

    response = client.get(url)
    assert response.status_code == 200
    assert b"Small Breeds" in response.data
    assert b"Large Breeds" in response.data
    assert b"HDB Approved" in response.data

def test_dogbreed():
    client = app.test_client()
    url = '/dog/Pug'

    response = client.get(url)
    assert response.status_code == 200
    assert b"It is not generally known that the Pug was the" in response.data

def test_cats():
    client = app.test_client()
    url = '/cats.html'

    response = client.get(url)
    assert response.status_code == 200
    assert b"Maine Coon" in response.data
    assert b"Exotic Shorthair" in response.data
    assert b"Ragdoll" in response.data

def test_catbreed():
    client = app.test_client()
    url = '/cat/Maine Coon'

    response = client.get(url)
    assert response.status_code == 200
    assert b"The Maine Coon is a large and sociable cat, hence" in response.data

def test_dogclassify():
    client = app.test_client()
    url = '/dogclassify.html'
    data = {'imagefile' : (os.path.join("./static/dogs/pug2.jpg"), 'static/dogs/pug2.jpg')}

    response = client.post(url, data=data, follow_redirects = True, content_type='multipart/form-data')
    assert b"Pug" in response.data
    assert response.status_code == 200

def test_dogclassify__failure__bad_request():
    client = app.test_client()
    url = '/dogclassify.html'
    data = {}

    response = client.post(url, data=data, follow_redirects = True, content_type='multipart/form-data')
    assert response.status_code == 400

def test_catclassify():
    client = app.test_client()
    url = '/catclassify.html'
    data = {'imagefile' : (os.path.join("./static/cats/persian1.jpg"), 'static/cats/persian1.jpg')}

    response = client.post(url, data=data, follow_redirects = True, content_type='multipart/form-data')
    assert b"Persian" in response.data
    assert response.status_code == 200

def test_catclassify__failure__bad_request():
    client = app.test_client()
    url = '/catclassify.html'
    data = {}

    response = client.post(url, data=data, follow_redirects = True, content_type='multipart/form-data')
    assert response.status_code == 400
