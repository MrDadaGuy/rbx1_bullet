FROM tensorflow/tensorflow:latest-gpu-py3
LABEL maintainer="veggiebenz@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
ENV HOME /root

RUN apt-get update && apt-get install -y lsb-release apt-utils curl wget sudo && apt-get clean all

# configure ROS repository
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -

# get base (novnc etc) dependencies
RUN apt-get install -y --no-install-recommends supervisor \
        openssh-server pwgen sudo vim-tiny nano \
        net-tools \
        lxde x11vnc x11vnc-data xvfb \
        gtk2-engines-murrine ttf-ubuntu-font-family \
        nginx \
        python-dev python3-dev build-essential 

# get project specific stuff
RUN apt-get update && apt-get install -y  \
        ffmpeg \
        ros-melodic-desktop-full \
        python-catkin-tools \
        ros-melodic-moveit-ros-planning \
        ros-melodic-moveit-ros-move-group \
        ros-melodic-moveit-ros-visualization \
        ros-melodic-moveit-planners-ompl \
        ros-melodic-moveit-fake-controller-manager \
	ros-melodic-rospy

# clean up APT 
RUN apt-get autoclean apt-get autoremove
# && rm -rf /var/lib/apt/lists/*

# install latest pip for python2 and python3
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && python3 get-pip.py

# get the volumes / content and python components for novnc
ADD docker-ubuntu-novnc/web /web/
ADD docker-ubuntu-novnc/noVNC /noVNC/
ADD docker-ubuntu-novnc/nginx.conf /etc/nginx/sites-enabled/default
ADD docker-ubuntu-novnc/startup.sh /
ADD docker-ubuntu-novnc/supervisord.conf /etc/supervisor/conf.d/
ADD docker-ubuntu-novnc/doro-lxde-wallpapers /usr/share/doro-lxde-wallpapers/
RUN /usr/local/bin/pip3 install -r /web/requirements.txt

# get our project specific python components  -- removed jupyter tensorflow-gpu (b/c its baked in) matplotlib pandas   
RUN pip3 install pybullet gym pyyaml rospkg PySide2


EXPOSE 6080 11311 9090 5900
WORKDIR /root
ENTRYPOINT ["/startup.sh"]

