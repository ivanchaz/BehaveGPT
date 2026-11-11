# BehaveGPT

![License](https://img.shields.io/badge/license-MIT-blue.svg)

This is a repo consisting of scripts and latest model on user behaviour modelling
used to predict what kind of next "action" user would do.

The method is based on the paper [BehaveGPT: A foundation model for large-scale user behaviour
modelling](https://arxiv.org/pdf/2505.17631). The credits on research goes to them.

Note: this repo is not owned nor endorsed by the author of the paper. As such, any error
or failure is not linked to them.

## Tech Stack

* Python

## Getting Started

### Prerequisites

Ensure you have the following installed locally:
* [Python](https://www.python.org/) (3.14.6 or higher)
* [Virtual Env](https://docs.python.org/3/library/venv.html) (optional, but recommended)

### Installation

* activate the virtual environment (optional)
    * `virtualenv -p python3 .venv`
    * `source .venv/bin/activate`
* install the third party libraries 
    * `pip install -r requirements.txt`
* download training data. at the moment, a modification to train.py is required
  to load the data. specifically, on the line 106 until 133.
* run the training script
    * `export PYTHONPATH="."; python src/train.py`

## Planned roadmap 

* Streamline a way to download data to be used for training
* Update the model to use time2vec
* train checkpoint
* separation between training and evaluate script

## FAQ

* how to get training data

Currently, the training data is obtained through private way using author's private data centre. As such, the script cannot be shared publicly. That being said, the data follow the following schema.

| Name           | Type   | Example Value                                                  |
|----------------|--------|----------------------------------------------------------------|
| userId         | string |                                                                |
| eventAction    | string |                                                                |
| globalEntityId | string | string identifier to identify where the event happened         |
| eventDate      | string | format "YYYY-MM-DD"                                            |
| eventTime      | string | format "hh:mm:ss.sss"                                          |
| brand          | string | the name of application which user's use to generate the event |

For example, the training data should be as follows (in JSON format)

```json
[{
  "userId": "12345",
  "eventAction": "page.visited",
  "globalEntityId": "France",
  "eventDate": "2026-09-14",
  "eventTime": "07:37:07.637",
  "brand": "uber or something else (empty string should also be fine)"
}]
```

One could obtain the public data of google analytics ((link)[https://console.cloud.google.com/marketplace/product/obfuscated-ga360-data/obfuscated-ga360-data?hl=en&project=dhh-perseus-live]) and normalize the data to conform with the schema.
