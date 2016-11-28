#!/bin/bash

#####################################
# Install script for Consumer Node  #
#####################################

abspath=`dirname "$(cd "${0%/*}" 2>/dev/null; echo "$PWD"/"${0##*/}")"`
parentdir=`dirname "$(cd "${abspath}/../" 2>/dev/null; echo "$PWD"/"${0##*/}")"`

whoami=`whoami`

# Verify properties directory exists
if [ ! -d "/deployments/edmunds/properties/common/" ];
then
    mkdir -p /deployments/edmunds/properties/common/
    chown webapps:webapps /deployments/edmunds/properties/common/
elif [ -d "/deployments/edmunds/properties/common/" ];
then
    chown webapps:webapps /deployments/edmunds/properties/common/
fi

#verify EPS directory exists
if [ ! -d "/deployments/eps/" ];
then
    mkdir -p /deployments/eps/
    chown webapps:webapps /deployments/eps/
elif [ -d "/deployments/eps/" ];
then
    chown webapps:webapps /deployments/eps/
fi

ln -sfn  ${parentdir} /deployments/eps/${project.artifactId}

#create sym link for jmxrmi agent jar file
pushd /deployments/eps/${project.artifactId}/lib
shortname="jmxrmi-agent"
realname=`ls $shortname*`
ln -sfn  $realname $shortname.jar
popd

chmod +x /deployments/eps/${project.artifactId}/*.sh
chmod +x /deployments/eps/${project.artifactId}/init/*
