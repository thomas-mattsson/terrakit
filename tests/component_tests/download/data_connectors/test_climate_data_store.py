# © Copyright IBM Corporation 2026
# SPDX-License-Identifier: Apache-2.0


import os
import pandas as pd
import pytest
import xarray as xr
from rasterio.crs import CRS

from terrakit import DataConnector
from terrakit.general_utils.exceptions import (
    TerrakitValidationError,
    TerrakitValueError,
)


class TestClimateDataStore:
    connector_type = "climate_data_store"
    # Mock data contains these 5 bands from the test zip file
    bands = ["fg10", "t2m", "tp", "u10", "v10"]

    @pytest.fixture
    def bbox(self):
        """Override default bbox with one large enough for ERA5 (0.25° resolution)."""
        # Kenya region: ~0.5° × 0.5° bbox (meets 0.25° minimum requirement)
        return [34.5, -0.5, 35.0, 0.0]

    def test_valid_data_connector(self):
        dc = DataConnector(connector_type=self.connector_type)
        assert dc.connector is not None

    def test_list_collections_climate_data_store(
        self,
        **kwargs,
    ):
        expected_collections = [
            "projections-cordex-domains-single-levels",
            "derived-era5-single-levels-daily-statistics",
        ]
        dc = DataConnector(connector_type=self.connector_type)
        collections = dc.connector.list_collections()
        assert collections == expected_collections

    def test__missing_credentials_cds(
        self,
        unset_evn_vars,
        start_date,
        bbox,
        reset_dot_env,
    ):
        """
        Test that find_data only runs if credentials are provided.
        """
        collection = "derived-era5-single-levels-daily-statistics"
        with pytest.raises(TerrakitValidationError, match="Error: Missing credentials"):
            dc = DataConnector(connector_type=self.connector_type)
            dc.connector.find_data(collection, start_date, start_date, bbox=bbox)

    def test_invalid_collection(self, start_date, bbox):
        """
        Test that an invalid collection raises a TerrakitValidationError.
        """
        collection = "invalid-collection"
        dc = DataConnector(connector_type=self.connector_type)
        with pytest.raises(TerrakitValueError, match="Invalid collection"):
            dc.connector.find_data(collection, start_date, start_date, bbox=bbox)

    @pytest.fixture
    def expected_dates_cds(self):
        dates = pd.date_range("2024-01-01", "2024-01-31").strftime("%Y-%m-%d").tolist()
        return dates

    @pytest.mark.parametrize(
        "collection",
        [
            ("derived-era5-single-levels-daily-statistics"),
            ("projections-cordex-domains-single-levels"),
        ],
    )
    def test_find_available_data_cds(
        self,
        collection,
        expected_dates_cds,
        start_date,
        end_date,
        bbox,
    ):
        dc = DataConnector(connector_type=self.connector_type)
        unique_dates, results = dc.connector.find_data(
            data_collection_name=collection,
            date_start=start_date,
            date_end=end_date,
            bbox=bbox,
            bands=self.bands,
        )
        assert unique_dates == expected_dates_cds

    @pytest.mark.parametrize(
        ("collection", "start_date", "end_date", "expected_dates_cds"),
        [
            (
                "derived-era5-single-levels-daily-statistics",
                "1949-01-01",
                "1949-01-02",
                ["1949-01-01, 1949-01-02"],
            ),
            (
                "projections-cordex-domains-single-levels",
                "2100-01-01",
                "2100-01-02",
                ["2100-01-01, 2100-01-02"],
            ),
        ],
    )
    def test_find_available_data_cds_start_data_given_constraints(
        self,
        collection,
        start_date,
        end_date,
        expected_dates_cds,
        bbox,
    ):
        """
        Test the find_data method with a given start date within the collection constraints.
        """
        dc = DataConnector(connector_type=self.connector_type)
        unique_dates, results = dc.connector.find_data(
            data_collection_name=collection,
            date_start=start_date,
            date_end=end_date,
            bbox=bbox,
            bands=self.bands,
        )
        assert unique_dates == [start_date, end_date]

    def test_find_available_data_box_too_small_for_era5_resolution(self, start_date):
        """
        Test that find_data raises error for bbox smaller than ERA5 grid resolution (0.25°).
        """
        collection = "derived-era5-single-levels-daily-statistics"
        # This bbox is only 0.02° x 0.02°, which is smaller than ERA5's 0.25° grid
        tiny_bbox = [-1.32, 51.06, -1.30, 51.08]

        dc = DataConnector(connector_type=self.connector_type)
        with pytest.raises(
            TerrakitValidationError,
            match="Bounding box too small for ERA5 data resolution",
        ):
            dc.connector.find_data(
                collection, start_date, start_date, bbox=tiny_bbox, bands=self.bands
            )

    def test_negative_longitude_conversion(
        self, mock_cds_client, start_date, bbox, save_file_dir, get_data_clean_up
    ):
        """
        Test that negative longitudes are automatically converted to 0-360° convention.

        CDS uses 0-360° longitude convention, but users typically use -180 to 180°.
        The connector should automatically convert negative longitudes.
        """
        collection = "derived-era5-single-levels-daily-statistics"
        # Use bbox with negative longitude (standard -180 to 180° convention)
        # Oxford, UK: -1.32° to -1.07° should be converted to 358.68° to 358.93°
        negative_lon_bbox = [-1.32, 51.70, -1.07, 51.95]
        save_file = f"{save_file_dir}/{self.connector_type}_{collection}.tif"

        dc = DataConnector(connector_type=self.connector_type)

        # This should work - the connector should convert negative longitudes
        data = dc.connector.get_data(
            data_collection_name=collection,
            date_start=start_date,
            date_end=start_date,
            bbox=negative_lon_bbox,
            bands=self.bands,
            save_file=save_file,
        )

        assert data is not None
        assert len(data) > 0

        # Verify the data was retrieved successfully
        assert isinstance(data, xr.DataArray)

    @pytest.mark.parametrize(
        "collection",
        [
            ("derived-era5-single-levels-daily-statistics"),
            # ("projections-cordex-domains-single-levels"),
        ],
    )
    def test_get_data_cds(
        self, mock_cds_client, collection, bbox, save_file_dir, get_data_clean_up
    ):
        """
        Test the get_data method.

        Note: The mock returns a zip file with 5 NetCDF files (one per variable),
        each containing 2 time steps (2025-01-01 and 2025-01-02).
        """
        date_start = "2025-01-01"
        date_end = "2025-01-02"
        save_file = f"{save_file_dir}/{self.connector_type}_{collection}.tif"
        dc = DataConnector(connector_type=self.connector_type)
        data = dc.connector.get_data(
            data_collection_name=collection,
            date_start=date_start,
            date_end=date_end,
            bbox=bbox,
            bands=self.bands,
            save_file=save_file,
        )
        assert data is not None
        assert len(data) > 0  # Check we got data

        assert isinstance(data, xr.DataArray)
        assert data.rio.crs == CRS.from_epsg(4326)

        # Mock data contains 5 bands (fg10, t2m, tp, u10, v10) - one per NetCDF file
        assert len(data.coords["band"]) == 5

        # Mock data contains 2 time steps (2025-01-01 and 2025-01-02)
        assert len(data.time) == 2

        # Check that files were created for the dates in the mock data
        assert os.path.exists(save_file.replace(".tif", "_2025-01-01.tif")) is True
        assert os.path.exists(save_file.replace(".tif", "_2025-01-02.tif")) is True

    def test_get_data_bbox_too_small_for_era5_resolution(
        self, mock_cds_client_bbox_error, start_date, save_file_dir
    ):
        """
        Test that get_data handles MARS error for bbox smaller than ERA5 grid resolution (0.25°).

        This test uses a mock that simulates the actual MARS error response when the bbox
        is too small. The error should be caught and converted to a TerrakitValidationError
        with a helpful message.
        """
        collection = "derived-era5-single-levels-daily-statistics"
        # This bbox is only 0.02° x 0.02°, which is smaller than ERA5's 0.25° grid
        tiny_bbox = [-1.32, 51.06, -1.30, 51.08]
        save_file = f"{save_file_dir}/{self.connector_type}_{collection}.tif"

        dc = DataConnector(connector_type=self.connector_type)
        with pytest.raises(
            TerrakitValidationError, match="CLIMATE DATA STORE REQUEST FAILED"
        ):
            dc.connector.get_data(
                data_collection_name=collection,
                date_start=start_date,
                date_end=start_date,
                bbox=tiny_bbox,
                bands=self.bands,
                save_file=save_file,
            )
