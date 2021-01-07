# Spotify_app

This repo contains the code for my Spotify project.
I created two scripts that fullfill the main goals of the project. *Other goals may be added*

**You can use the provided 'Template-...py' scripts for your own playlists/projects. Just remember to modify everything I marked accordingly.You can change them according to your needs or even build small apps that can be used without much coding knowledge**


The project will be updated from time to time as I intend to use and improve the python scripts over time. They will however always solve the same key issues.

## DUO.py
The first part of the project was the creation of a better Duos Playlist for my girlfriend and me by using our data obtained through the Spotify API.
We were not satisfied with the "Duo Mix" Playlist Spotify provided through our Spotify Duo Subscription. The artists seemed to be "correct" but the songs were not. 
It hardly offered any songs we actually both listened to and felt more like a wild guessing game based on our tastes than an actual combination of shared favorites and good recommendations.
Which is why I created a python script that creates a playlist that does just that: enter DUO.py
So far we are satiesfied with the results, and any ideas for improvement (such as leveraging machine learning) will be implemented soon.

## PlaylistBuddy
The spotify recommendations below each playlist are a nice feature, but depending on the playlist they sometimes feel a little unrefined.
I therefore took the liberty to try to refine them. The PlaylistBuddy will add 1-20 new songs to your playlist, based on spotifies recommendations and either their similarity to the existing songs in a given playlist, their similarity to the "average" song of a playlist, some manually selected feature filters or all of the above.

