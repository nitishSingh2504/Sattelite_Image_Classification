# =========================================================
# GOOGLE COLAB DASHBOARD (FINAL + PIXEL COUNT ADDED)
# =========================================================

from google.colab import output
output.enable_custom_widget_manager()

!pip install geemap earthengine-api ipywidgets rasterio --quiet

import ee
import geemap
import ipywidgets as widgets
from IPython.display import display, clear_output

# ---------------------------------------------------------
# GEE INIT
# ---------------------------------------------------------
PROJECT_ID = "ee-aparnaciirs"

try:
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine initialized")
except:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine authenticated")

# ---------------------------------------------------------
# 🔗 MAP SYNC FUNCTION
# ---------------------------------------------------------
def sync_maps(maps):

    def on_center_change(change):
        for m in maps:
            if m is not change["owner"]:
                m.center = change["new"]

    def on_zoom_change(change):
        for m in maps:
            if m is not change["owner"]:
                m.zoom = change["new"]

    for m in maps:
        m.observe(on_center_change, "center")
        m.observe(on_zoom_change, "zoom")

# ---------------------------------------------------------
# CLOUD MASKS
# ---------------------------------------------------------
def mask_sentinel2_clouds(image):
    qa = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(
           qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)

def mask_landsat_clouds(image):
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(
           qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)

# ---------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------
def get_collection(sensor, roi, start_date, end_date):

    if sensor == "Sentinel-2":
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(roi)
               .filterDate(start_date, end_date)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
               .map(mask_sentinel2_clouds))

        bands = ["B2", "B3", "B4", "B8"]
        vis = {"bands": ["B4","B3","B2"], "min":0, "max":0.3}
        scale = 10

    else:
        col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
               .filterBounds(roi)
               .filterDate(start_date, end_date)
               .map(mask_landsat_clouds))

        bands = ["SR_B2","SR_B3","SR_B4","SR_B5"]
        vis = {"bands": ["SR_B4","SR_B3","SR_B2"], "min":5000, "max":15000}
        scale = 30

    return col.median().clip(roi), bands, vis, scale

# ---------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------
def run_unsupervised_classification(image, bands, roi, clusters):
    training = image.select(bands).sample(
        region=roi, scale=30, numPixels=5000)
    clusterer = ee.Clusterer.wekaKMeans(clusters).train(training)
    return image.select(bands).cluster(clusterer)

# ---------------------------------------------------------
# PIXEL COUNT FUNCTION
# ---------------------------------------------------------
def get_class_pixel_count(classified_img, roi, scale):

    stats = classified_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=roi,
        scale=scale,
        maxPixels=1e10
    )

    return stats.getInfo()

def format_counts(hist_dict):
    if not hist_dict:
        return "No data"

    hist = list(hist_dict.values())[0]
    return "<br>".join([f"Class {k}: {v} pixels" for k, v in hist.items()])

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------
polygon_input = widgets.Textarea(
    value="72.85,19.05; 72.95,19.05; 72.95,19.15; 72.85,19.15",
    description="ROI:",
    layout=widgets.Layout(width="400px", height="100px")
)

start_date_input = widgets.Text(value="2025-01-01", description="Start:")
end_date_input   = widgets.Text(value="2025-03-31", description="End:")

cluster_slider = widgets.IntSlider(
    value=5, min=3, max=10, description="Clusters")

run_button = widgets.Button(
    description="Run", button_style="success")

output_box = widgets.Output()
map_output = widgets.Output()

# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def run_dashboard(b):

    global Map1, Map2, Map3, Map4

    with output_box:
        clear_output()
        print("Processing...")

    try:
        # ROI
        coords = []
        for pair in polygon_input.value.split(";"):
            lon, lat = map(float, pair.strip().split(","))
            coords.append([lon, lat])
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        roi = ee.Geometry.Polygon([coords])

        # DATA
        s_img, s_bands, s_vis, s_scale = get_collection(
            "Sentinel-2", roi,
            start_date_input.value,
            end_date_input.value)

        l_img, l_bands, l_vis, l_scale = get_collection(
            "Landsat", roi,
            start_date_input.value,
            end_date_input.value)

        # CLASSIFICATION
        s_cls = run_unsupervised_classification(
            s_img, s_bands, roi, cluster_slider.value)

        l_cls = run_unsupervised_classification(
            l_img, l_bands, roi, cluster_slider.value)

        # PIXEL COUNTS
        s_counts = get_class_pixel_count(s_cls, roi, s_scale)
        l_counts = get_class_pixel_count(l_cls, roi, l_scale)

        # MAPS
        Map1 = geemap.Map(center=[19.07,72.87], zoom=10)
        Map2 = geemap.Map(center=[19.07,72.87], zoom=10)
        Map3 = geemap.Map(center=[19.07,72.87], zoom=10)
        Map4 = geemap.Map(center=[19.07,72.87], zoom=10)

        for m in [Map1,Map2,Map3,Map4]:
            m.add_basemap("OpenStreetMap")
            m.centerObject(roi,11)
            m.layout.height="400px"
            m.layout.width="500px"

        sync_maps([Map1, Map2, Map3, Map4])

        # TRUE COLOR
        Map1.addLayer(s_img, s_vis, "Sentinel TCC")
        Map2.addLayer(l_img, l_vis, "Landsat TCC")

        # CLASS COLORS
        palette = [
            "red","green","blue","yellow","orange",
            "purple","cyan","magenta","brown","gray"
        ]

        vis_params = {
            "min": 0,
            "max": cluster_slider.value - 1,
            "palette": palette[:cluster_slider.value]
        }

        Map3.addLayer(s_cls, vis_params, "Sentinel Class")
        Map4.addLayer(l_cls, vis_params, "Landsat Class")

        # LEGEND
        legend = {f"Class {i}": palette[i] for i in range(cluster_slider.value)}
        Map3.add_legend(title="Classes", legend_dict=legend)
        Map4.add_legend(title="Classes", legend_dict=legend)

        # PIXEL COUNT DISPLAY
        sentinel_text = widgets.HTML(
            f"<b>Sentinel Pixel Count</b><br>{format_counts(s_counts)}"
        )

        landsat_text = widgets.HTML(
            f"<b>Landsat Pixel Count</b><br>{format_counts(l_counts)}"
        )

        # DISPLAY
        with map_output:
            clear_output()

            display(widgets.VBox([

                widgets.HBox([
                    widgets.VBox([widgets.HTML("<b>Sentinel TCC</b>"), Map1]),
                    widgets.VBox([widgets.HTML("<b>Landsat TCC</b>"), Map2])
                ]),

                widgets.HBox([
                    widgets.VBox([widgets.HTML("<b>Sentinel Class</b>"), Map3, sentinel_text]),
                    widgets.VBox([widgets.HTML("<b>Landsat Class</b>"), Map4, landsat_text])
                ])

            ]))

        with output_box:
            print("✅ Done")

    except Exception as e:
        with output_box:
            print("❌ Error:", e)

run_button.on_click(run_dashboard)

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
left_panel = widgets.VBox([
    widgets.HTML("<h3>GEE Dashboard</h3>"),
    polygon_input,
    start_date_input,
    end_date_input,
    cluster_slider,
    run_button,
    output_box
])

right_panel = widgets.VBox([map_output])

display(widgets.HBox([left_panel, right_panel]))