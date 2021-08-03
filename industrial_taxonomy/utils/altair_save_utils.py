# Scripts to save altair charts

from altair_saver import save
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import os

import industrial_taxonomy

project_dir = industrial_taxonomy.project_dir

FIG_PATH = f"{project_dir}/figures"

# Checks if the right paths exist and if not creates them when imported
os.makedirs(f"{FIG_PATH}/png", exist_ok=True)
os.makedirs(f"{FIG_PATH}/html", exist_ok=True)


def google_chrome_driver_setup():
    # Set up the driver to save figures as png
    driver = webdriver.Chrome(ChromeDriverManager().install())
    return driver


def save_altair(fig, name, driver, path=FIG_PATH):
    """Saves an altair figure as png and html
    Args:
        fig: altair chart
        name: name to save the figure
        driver: webdriver
        path: path to save the figure
    """
    save(
        fig,
        f"{path}/png/{name}.png",
        method="selenium",
        webdriver=driver,
        scale_factor=5,
    )
    fig.save(f"{path}/html/{name}.html")

def altair_text_resize(chart, sizes=[12, 14]):

    ch = chart.configure_axis(
        labelFontSize=sizes[0], titleFontSize=sizes[1]
    ).configure_legend(labelFontSize=sizes[0], titleFontSize=sizes[1])
    return ch