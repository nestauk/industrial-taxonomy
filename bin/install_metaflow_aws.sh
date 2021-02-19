#!/bin/bash

# Fetch research daps key
aws s3 cp s3://nesta-production-config/research_daps.key . &&\
# Clone research daps
# cd /tmp &&\
 git clone git@github.com:nestauk/research_daps.git &&\
 cd research_daps &&\
# Unencrypt research daps
 git-crypt unlock ../research_daps.key &&\
# Copy metaflow config, TODO: backup an existing config
 mkdir -p ~/.metaflowconfig &&\
 cp research_daps/config/metaflowconfig/config.json ~/.metaflowconfig/.

