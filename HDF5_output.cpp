#include "HDF5_output.h"
#include <iostream>

// Using the HDF5 C++ namespace
using namespace H5;

// The function saveRankData creates (or overwrites) an HDF5 file and writes each tile’s data
// It assumes that the type Grid is defined (in Auxiliar_functions.h) as:
// struct Grid { double Ex, Ey, Ez, Bx, By, Bz };
void saveRankData(const RankInfo &info, int guard, const std::string &filename) {
    try {
        // Create a new HDF5 file (overwrite if it exists)
        H5File file(filename, H5F_ACC_TRUNC);

        // Define a compound HDF5 type matching the Grid structure
        CompType gridType(sizeof(Grid));
        gridType.insertMember("Ex", HOFFSET(Grid, Ex), PredType::NATIVE_DOUBLE);
        gridType.insertMember("Ey", HOFFSET(Grid, Ey), PredType::NATIVE_DOUBLE);
        gridType.insertMember("Ez", HOFFSET(Grid, Ez), PredType::NATIVE_DOUBLE);
        gridType.insertMember("Bx", HOFFSET(Grid, Bx), PredType::NATIVE_DOUBLE);
        gridType.insertMember("By", HOFFSET(Grid, By), PredType::NATIVE_DOUBLE);
        gridType.insertMember("Bz", HOFFSET(Grid, Bz), PredType::NATIVE_DOUBLE);

        // Loop over each tile in the rank and write its data to the file
        for (size_t i = 0; i < info.tiles.size(); i++) {
            const Tile &tile = info.tiles[i];

            // Compute dimensions for the tile grid (including guard cells)
            int totalX = tile.info.nx + 2 * guard;
            int totalY = tile.info.ny + 2 * guard;
            hsize_t dims[2] = { static_cast<hsize_t>(totalY), static_cast<hsize_t>(totalX) };

            // Create a group for this tile; group name based on its globalID
            std::string groupName = "/Tile_" + std::to_string(tile.info.globalID);
            Group tileGroup = file.createGroup(groupName);

            // Create a dataspace for the dataset (2D array of Grid elements)
            DataSpace dataspace(2, dims);

            // Create the dataset within the group
            // Here the dataset is named "fields" and stores the Grid data
            DataSet dataset = tileGroup.createDataSet("fields", gridType, dataspace);

            // Write the tile's grid data (assumed to be stored in a contiguous vector)
            dataset.write(&tile.grid[0], gridType);

            // Save the tile's row index
            {
                int tileRow = tile.info.tileRow;
                DataSpace attrSpace(H5S_SCALAR);
                Attribute attrTileRow = tileGroup.createAttribute("tileRow", PredType::NATIVE_INT, attrSpace);
                attrTileRow.write(PredType::NATIVE_INT, &tileRow);
            }
            // Save the tile's column index
            {
                int tileCol = tile.info.tileCol;
                DataSpace attrSpace(H5S_SCALAR);
                Attribute attrTileCol = tileGroup.createAttribute("tileCol", PredType::NATIVE_INT, attrSpace);
                attrTileCol.write(PredType::NATIVE_INT, &tileCol);
            }
            // Save the rank that currently owns the tile
            {
                int currentRank = tile.info.currentRank;
                DataSpace attrSpace(H5S_SCALAR);
                Attribute attrRank = tileGroup.createAttribute("currentRank", PredType::NATIVE_INT, attrSpace);
                attrRank.write(PredType::NATIVE_INT, &currentRank);
            }
        }
    }
    catch (FileIException &error) {
        error.printErrorStack();
    }
    catch (DataSetIException &error) {
        error.printErrorStack();
    }
    catch (DataSpaceIException &error) {
        error.printErrorStack();
    }
}
