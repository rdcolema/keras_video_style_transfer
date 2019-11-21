# Keras Video Style Transfer

<p> An implementation of <a href="https://github.com/titu1994/Neural-Style-Transfer">neural style transfer</a> applied to video with Keras</p>

<h3>Dependencies</h3>
<ul>
<li>keras (with theano backend)</li>
<li>numpy</li>
<li>scipy</li>
<li>cv2</li>
<li>matplotlib (for video scripts)</li>
<li>moviepy (for video scripts)</li>
<li>ffmpeg (for video scripts)</li>
<li>Cuda (technically optional, but required for GPU)</li>
</ul>

<h3>Instructions</h3>
<ul>
<li>Requires individual frames of the video to be styled placed in the <em>base_image_path</em> directory and labeled appropriately before running. Then run <em>keras_video_style.py</em>.</li>
</ul>

<h3>Example Output</h3>

Style            |  Result
:-------------------------:|:-------------------------:
![](https://github.com/rdcolema/keras-neural-style-transfer/blob/master/examples/style_images/fire.jpg)  |  ![](https://media.giphy.com/media/3oriNKT125H9nllol2/source.gif)
![](https://github.com/rdcolema/keras-neural-style-transfer/blob/master/examples/style_images/kandinsky.jpg)  |  ![](https://media.giphy.com/media/3o7TKO1r6OdGpZz4m4/source.gif)
