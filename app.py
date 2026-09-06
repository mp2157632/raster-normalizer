from flask import Flask, render_template, request, send_file
import os
import uuid
import numpy as np
import rasterio
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "tif", "tiff"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/normalize", methods=["POST"])
def normalize():

    if "file" not in request.files:
        return {"error": "No file was uploaded."}, 400

    file = request.files["file"]

    if file.filename == "":
        return {"error": "Please select a file."}, 400

    if not allowed_file(file.filename):
        return {
            "error": "Invalid file format. Please upload JPG or TIFF."
        }, 400

    extension = file.filename.rsplit(".", 1)[1].lower()

    unique_name = str(uuid.uuid4())
    input_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name + "." + extension
    )

    file.save(input_path)

    try:

        # -------------------------
        # JPG / JPEG processing
        # -------------------------
        if extension in {"jpg", "jpeg"}:

            image = Image.open(input_path).convert("RGB")
            array = np.array(image)

            normalized = normalize_array(array)

            output_path = os.path.join(
                OUTPUT_FOLDER,
                unique_name + "_normalized.tif"
            )

            height, width, bands = normalized.shape

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=bands,
                dtype="uint8"
            ) as dst:

                for band in range(bands):
                    dst.write(
                        normalized[:, :, band],
                        band + 1
                    )

        # -------------------------
        # TIFF processing
        # -------------------------
        else:

            with rasterio.open(input_path) as src:

                data = src.read()

                normalized = normalize_array(data)

                output_path = os.path.join(
                    OUTPUT_FOLDER,
                    unique_name + "_normalized.tif"
                )

                profile = src.profile.copy()

                profile.update(
                    dtype="uint8",
                    count=src.count,
                    compress="lzw"
                )

                with rasterio.open(
                    output_path,
                    "w",
                    **profile
                ) as dst:

                    dst.write(normalized)

        # Delete uploaded temporary file
        os.remove(input_path)

        return {
            "success": True,
            "message": "Normalization completed successfully.",
            "download_url": "/download/" + os.path.basename(output_path)
        }

    except Exception as e:

        if os.path.exists(input_path):
            os.remove(input_path)

        return {
            "error": "Processing failed: " + str(e)
        }, 500


def normalize_array(data):

    data = data.astype(np.float32)

    minimum = np.nanmin(data)
    maximum = np.nanmax(data)

    if maximum == minimum:
        normalized = np.zeros_like(data)
    else:
        normalized = (
            (data - minimum)
            / (maximum - minimum)
            * 255
        )

    normalized = np.nan_to_num(
        normalized,
        nan=0,
        posinf=255,
        neginf=0
    )

    normalized = np.clip(
        normalized,
        0,
        255
    )

    return normalized.astype(np.uint8)


@app.route("/download/<filename>")
def download(filename):

    file_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    if not os.path.exists(file_path):
        return "File not found.", 404

    return send_file(
        file_path,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)