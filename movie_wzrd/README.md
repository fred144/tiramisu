# Rendering

needs sim_scraper.py to be ran to get the logSFCs.

point it to a rendering_dir, and name it movie_name.mp4

And change the framerate N, usually around 20 - 30 fps. 

ffmpeg -framerate N -pattern_type glob -i 'rendering_dir/*.png'   -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" movie_name.mp4