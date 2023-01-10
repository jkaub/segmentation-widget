import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import Button, Dropdown, HBox, VBox
from skimage.draw import polygon

class SegmentWidget:
    """
    A widget for image segmentation.
    
    This class allows the user to segment an image by creating a polygon on the image
    and generating a binary mask from the polygon. The user can navigate through a set
    of images and masks and create a binary mask for each one. The class displays the 
    original image and the corresponding mask in a plotly figure widget,
    and support user interaction to draw the polygon over the image and update the 
    mask accordingly.

    Example:
    ```
    widget = SegmentWidget('path/to/imgs', 'path/to/masks')
    widget.show()
    ```
    """    
    def __init__(self, path_imgs, path_masks):

        """
        Initialize the SegmentWidget.
        
        Args:
        path_imgs (str): The path to the directory containing the images in numpy format
        path_masks (str): The path to the directory containing the masks in numpy format
        
        Attributes:
        _path_imgs (str): The path to the directory containing the images
        _path_masks (str): The path to the directory containing the masks
        _ids (list): A list of image filenames
        _current_id (str): The current image id
        _polygon_coordinates (list): A list of coordinates of the polygon used to segment the images
        _current_img (np.ndarray): The currently loaded image
        _current_mask (np.ndarray): The currently loaded mask
        _intermediate_mask (np.ndarray): The mask that is currently being edited by the user
        _image_fig (plotly.graph_objs._figurewidget.FigureWidget): The figure widget for the image
        _mask_fig (plotly.graph_objs._figurewidget.FigureWidget): The figure widget for the mask
        """        
        
        self._path_imgs = path_imgs

        #init the path to store the masks if it does not exist
        if not os.path.exists(path_masks):
            os.mkdir(path_masks)

        self._path_masks = path_masks
        self._ids = sorted([os.path.splitext(e)[0] for e in  os.listdir(path_imgs)])
        self._current_id = self._ids[0]
        self._polygon_coordinates = []
        #Initiation of all the components
        self._initialize_widget()
        
    def _load_images(self):
        '''This method will be used to load image and mask when we select another image'''
        img_path = os.path.join(self._path_imgs,f"{self._current_id}.npy")
        self._current_img = np.load(img_path) # This is a 3D numpy array (256,256,3)
        h,w, _ = self._current_img.shape
        
        #There is not always a mask saved. When no mask is saved, we create an empty one.
        mask_path = os.path.join(self._path_masks,f"{self._current_id}.npy")
        if os.path.exists(mask_path):
            self._current_mask = np.load(mask_path)  # This is a 2D numpy array (256,256)
        else:
            self._current_mask = np.zeros((h,w))
        
        self._intermediate_mask = self._current_mask.copy()
        
    def _gen_mask_from_polygon(self):
        '''This function set to 2 the values inside the polygon defined by the list of points provided'''
        h,w = self._current_mask.shape
        new_mask = np.zeros((h,w), dtype=int)
        rr, cc = polygon([e[0] for e in self._polygon_coordinates], 
                         [e[1] for e in self._polygon_coordinates], shape=new_mask.shape)

        self._intermediate_mask = self._current_mask.copy()
        self._intermediate_mask[rr,cc]=2
    
    def _on_click_figure(self, trace, points, state):
        '''Callback for clicking on the figure. At each click, the coordinates of the click are stored in the polygon coordinates
           and the figure is displayed again
        '''
        #Retrieve coordinates of the clicked point
        i,j = points.point_inds[0]
        #Add the point to the list of points
        self._polygon_coordinates.append((i,j))
        
        if len(self._polygon_coordinates)>2:
            self._gen_mask_from_polygon()
            with self._image_fig.batch_update():
                self._image_fig.data[1].z = self._intermediate_mask      
        
    def _initialize_figures(self):
        '''This function is called to initialize the figure and its callback'''
        self._image_fig = go.FigureWidget()
        self._mask_fig = go.FigureWidget()
        
        self._load_images()
        #We use plotly express to generate the RGB image from the 3D array loaded
        img_trace = px.imshow(self._current_img).data[0]
        #We use plotly HeatMap for the 2D mask array
        mask_trace = go.Heatmap(z=self._current_mask, showscale=False, zmin=0, zmax=1)
        
        #Add the traces to the figures
        self._image_fig.add_trace(img_trace)
        self._image_fig.add_trace(mask_trace)
        self._mask_fig.add_trace(mask_trace)
        
        #A bit of chart formating
        self._image_fig.data[1].opacity = 0.3 #make the mask transparent on image 1
        self._image_fig.data[1].zmax = 2 #the overlayed mask above the image can have values in range 0..2
        self._image_fig.update_xaxes(visible=False)
        self._image_fig.update_yaxes(visible=False)
        self._image_fig.update_layout(margin={"l": 10, "r": 10, "b": 10, "t": 50}, 
                                      title = "Define your Polygon Here",
                                      title_x = 0.5, title_y = 0.95)
        self._mask_fig.update_layout(yaxis=dict(autorange='reversed'), margin={"l": 0, "r": 10, "b": 10, "t": 50},)
        self._mask_fig.update_xaxes(visible=False)
        self._mask_fig.update_yaxes(visible=False)
    
        self._image_fig.data[-1].on_click(self._on_click_figure)

    def _callback_save_button(self, button):
        '''This callback save the current mask and reset the polygon coordinates to start a new label'''
        self._current_mask[self._intermediate_mask==2]=1
        self._current_mask[self._intermediate_mask==0]=0
        mask_path = os.path.join(self._path_masks,f"{self._current_id}.npy")
        np.save(mask_path,self._current_mask)
        self._intermediate_mask = self._current_mask.copy()
        with self._image_fig.batch_update():
            self._image_fig.data[1].z = self._current_mask     
        with self._mask_fig.batch_update():
            self._mask_fig.data[0].z = self._current_mask
        self._polygon_coordinates = []
            
    def _build_save_button(self):
        self._save_button = Button(description="Save Configuration")
        self._save_button.on_click(self._callback_save_button)
        
    def _callback_delete_current_config_button(self, button):
        '''This callback reset the intermediate_mask to the currently saved mask and refresh the figure'''
        self._intermediate_mask = self._current_mask.copy()
        with self._image_fig.batch_update():
            self._image_fig.data[1].z = self._intermediate_mask
        self._polygon_coordinates = []
        
    def _build_delete_current_config_button(self):
        self._delete_current_config_button = Button(description="Delete Current Mask")
        self._delete_current_config_button.on_click(self._callback_delete_current_config_button)
        
    def _callback_delete_all_button(self, button):
        '''This callback reset the intermediate_mask to 0 and refresh the figure'''
        self._intermediate_mask[:] = 0
        with self._image_fig.batch_update():
            self._image_fig.data[1].z = self._intermediate_mask
        self._polygon_coordinates = []
        
    def _build_delete_all_button(self):
        self._delete_all_button = Button(description="Delete All Mask")
        self._delete_all_button.on_click(self._callback_delete_all_button)
        
    def _callback_dropdown(self, change):
        '''This callback is used to navigate through the different images'''
        #Set the new id to the new dropdown value
        self._current_id = change['new']
        
        #Load the new image and the new mask, we already have a method to do this
        self._load_images()
        
        img_trace = px.imshow(self._current_img).data[0]

        #Update both figure
        with self._image_fig.batch_update():
            #Update the trace 0 and the trace 1 containing respectively
            #the image and the mask
            self._image_fig.data[0].source = img_trace.source
            self._image_fig.data[1].z = self._current_mask
        
        with self._mask_fig.batch_update():
            self._mask_fig.data[0].z = self._current_mask
            
        #Reset the list of coordinates used to store current work in progress
        self._polygon_coordinates = []
        
    def _build_dropdown(self):
        #The ids are passed as option for the dropdown
        self._dropdown = Dropdown(options = self._ids)
        self._dropdown.observe(self._callback_dropdown, names="value")            
            
    def _initialize_widget(self):
        '''Function called during the init phase to initialize all the components
           and build the widget layout
        '''
        
        #Initialize the components
        self._initialize_figures()
        self._build_save_button()
        self._build_delete_current_config_button()
        self._build_delete_all_button()
        self._build_dropdown()
        
        #Build the layout
        buttons_widgets = HBox([self._save_button,self._delete_current_config_button,self._delete_all_button])
        figure_widgets = HBox([self._image_fig, self._mask_fig])
        self.widget = VBox([self._dropdown, buttons_widgets, figure_widgets])
    
    def display(self):
        display(self.widget)