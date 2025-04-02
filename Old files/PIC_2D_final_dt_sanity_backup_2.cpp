#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "Auxiliar_functions.h"  // Contains all helper functions and their definitions
#include "Field_update.h"        // External field update routines for evolving electromagnetic fields
#include "HDF5_output.h"         // Function to save tile data to an HDF5 file

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine grid decomposition for the given number of ranks
    int R, C; // Number of rows (R) and columns (C) in the rank grid
    findBestGrid(size, R, C);
    int rankRow = rank / C;
    int rankCol = rank % C;

    // Each rank initially holds 9 tiles (arranged in a 3x3 grid)
    int numTiles = 9;
    int tileRows, tileCols;
    findBestTileGrid(numTiles, tileRows, tileCols);

    // Initialize the RankInfo for the current rank
    RankInfo info;
    info.rank = rank;
    info.rankRow = rankRow;
    info.rankCol = rankCol;
    info.tileRows = tileRows;
    info.tileCols = tileCols;
    info.tiles.resize(numTiles);

    // Total global domain of tiles is (R * tileRows) x (C * tileCols)
    int globalTileRows = R * tileRows;
    int globalTileCols = C * tileCols;
    int totalGlobalTiles = globalTileRows * globalTileCols;

    // Ownership array: which rank owns each tile (initially, each tile is owned by its local rank)
    vector<int> owner(totalGlobalTiles, -1);

    // Simulation Setup
    double box_x = 10.0, box_y = 10.0;                  // Physical box dimensions [c/\omega_p]
    int nx = 180, ny = 180;                             // Total number of grid cells in x and y
    double dx = box_x / nx;                             // Spatial step size in x [c/\omega_p]
    double dy = box_y / ny;                             // Spatial step size in y [c/\omega_p]
    int guard = 2;                                      // Number of guard cells

    int interior_nx = nx / (C * tileCols);              // Number of interior cells in x for each tile
    int interior_ny = ny / (R * tileRows);              // Number of interior cells in y for each tile

    double sim_time = 10;                               // Total simulation time [\omega_p^{-1}]
    double dt = 0.99 * dx;                              // Time step size (respecting the CFL condition)
    int total_steps = static_cast<int>(sim_time / dt);  // Number of simulation steps
    int save_frequency = 10;                            // Frequency of saving simulation data

    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t];
        int tRow = t / tileCols; // Tile row index within the rank
        int tCol = t % tileCols; // Tile column index within the rank
        int gRow = tileGlobalRow(rankRow, tileRows, tRow); // Global row index of the tile
        int gCol = tileGlobalCol(rankCol, tileCols, tCol); // Global column index of the tile

        tile.info.globalID = getGlobalID(gRow, gCol, globalTileCols);
        tile.info.tileRow = gRow;
        tile.info.tileCol = gCol;
        tile.info.nx = interior_nx;
        tile.info.ny = interior_ny;
        tile.info.currentRank = rank;

        int totalX = interior_nx + 2 * guard; 
        int totalY = interior_ny + 2 * guard;
        tile.grid.resize(totalX * totalY);

        // Compute global offsets for each tile
        int global_x_offset = tile.info.tileCol * interior_nx;
        int global_y_offset = (globalTileRows - 1 - tile.info.tileRow) * interior_ny;

        //cout << "Rank " << rank << " Tile " << t << " Global ID: " << tile.info.globalID
        //     << " Global X Offset: " << global_x_offset << " Global Y Offset: " << global_y_offset << endl;

        // Initialize the fields for each tile
        // Wave parameters for initial Ex field:
        double A = 0.1;                     // Wave amplitude
        double kx = 5 * 2 * M_PI / box_x;   // Wave number in x
        double ky = 0;                      // Wave number in y
        for (int j = 0; j < totalY; j++) {
            for (int i = 0; i < totalX; i++) {
                // Compute physical coordinates:
                double x_Ex = (global_x_offset + i - guard + 0.5) * dx;
                double y_Ex = (global_y_offset + j - guard + 0) * dy;
                int idx = j * totalX + i;
                tile.grid[idx].Ex = global_y_offset + j ; //* sin(kx * x_Ex + ky * y_Ex);
                tile.grid[idx].Ey = 0.0; //* sin(kx * x_Ey + ky * y_Ey);
                tile.grid[idx].Ez = 0.0;
                tile.grid[idx].Bx = 0.0;
                tile.grid[idx].By = 0.0;
                tile.grid[idx].Bz = 0.0;
            }
        }
        owner[tile.info.globalID] = rank;
    }

    // Synchronize ownership info across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Save the initial state of the simulation
    if (rank == 0) std::cout << "Saving initial fields before time-stepping." << std::endl;
    std::string filename = "Simulation/Fields/fields_rank_" + std::to_string(rank) + "_step_0.h5";
    saveRankData(info, guard, filename);

    // ----------------------- Time-Stepping Simulation Loop -----------------------
    for (int step = 1; step < total_steps; step++) { // Start at step 1 (after the initial state)
        // 1) Field Updates:
        // For each tile, extract current field values into temporary arrays,
        // update the fields using external routines, and merge the updated fields back
        for (int i = 0; i < info.tiles.size(); i++) {
            Tile &tile = info.tiles[i];
            int totalX = tile.info.nx + 2 * guard;
            int totalY = tile.info.ny + 2 * guard;
            int totalCells = totalX * totalY;

            // Separate arrays for the electric field (Efull) and magnetic field (Bhalf)
            vector<GridE> Efull(totalCells);
            vector<GridB> BhalfOld(totalCells);
            for (int idx = 0; idx < totalCells; idx++) {
                Efull[idx].Ex = tile.grid[idx].Ex;
                Efull[idx].Ey = tile.grid[idx].Ey;
                Efull[idx].Ez = tile.grid[idx].Ez;
                BhalfOld[idx].Bx = tile.grid[idx].Bx;
                BhalfOld[idx].By = tile.grid[idx].By;
                BhalfOld[idx].Bz = tile.grid[idx].Bz;
            }
            vector<GridB> BhalfNew(totalCells);

            // Update the electromagnetic fields using external routines:
            updateBhalf(Efull, BhalfOld, BhalfNew, interior_nx, interior_ny, guard, dt, dx, dy);
            updateEfull(Efull, BhalfNew, interior_nx, interior_ny, guard, dt, dx, dy);

            // Merge the updated fields back into the tile's grid
            for (int idx = 0; idx < totalCells; idx++) {
                tile.grid[idx].Ex = Efull[idx].Ex;
                tile.grid[idx].Ey = Efull[idx].Ey;
                tile.grid[idx].Ez = Efull[idx].Ez;
                tile.grid[idx].Bx = BhalfNew[idx].Bx;
                tile.grid[idx].By = BhalfNew[idx].By;
                tile.grid[idx].Bz = BhalfNew[idx].Bz;
            }
        }

        // 2) Guard-cell Communication:
        // Each tile exchanges its guard regions with neighboring tiles
        int localTileCount = info.tiles.size();
        int totalComm = localTileCount * 8;
        vector<MPI_Request> requests(totalComm * 2);
        int reqCount = 0;
        vector< vector< vector<Grid> > > recvBuffers(localTileCount, vector< vector<Grid> >(8));

        // Post non-blocking receives for guard cell data
        for (int i = 0; i < localTileCount; i++) {
            Tile &tile = info.tiles[i];
            int gID = tile.info.globalID;
            for (int d = 0; d < 8; d++) {
                int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                int nbrRank = owner[nbrGID];
                int nbrDirection = opposite[d];
                int tag = computeTag(nbrGID, nbrDirection);
                int bufferSize = 0;
                if (d == 0 || d == 1)
                    bufferSize = interior_ny * guard;
                else if (d == 2 || d == 3)
                    bufferSize = interior_nx * guard;
                else
                    bufferSize = guard * guard;
                recvBuffers[i][d].resize(bufferSize);
                MPI_Irecv(&recvBuffers[i][d][0], bufferSize * sizeof(Grid), MPI_BYTE, nbrRank, tag, MPI_COMM_WORLD, &requests[reqCount]);
                reqCount++;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Post non-blocking sends for guard cell data
        for (int i = 0; i < localTileCount; i++) {
            Tile &tile = info.tiles[i];
            int gID = tile.info.globalID;
            for (int d = 0; d < 8; d++) {
                int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                int nbrRank = owner[nbrGID];
                int tagSend = computeTag(gID, d);
                vector<Grid> sendBuffer = packSendBuffer(tile, d, guard);
                MPI_Isend(&sendBuffer[0], sendBuffer.size() * sizeof(Grid), MPI_BYTE, nbrRank, tagSend, MPI_COMM_WORLD, &requests[reqCount]);
                reqCount++;
            }
        }

        MPI_Waitall(reqCount, &requests[0], MPI_STATUSES_IGNORE);

        // Update guard regions for each tile with the received data
        for (int i = 0; i < localTileCount; i++) {
            Tile &tile = info.tiles[i];
            for (int d = 0; d < 8; d++) {
                updateGuardRegion(tile, d, guard, recvBuffers[i][d]);
            }
        }

        // 3) Dynamic Tile Migration:
        // At a specific time step (here step==5 and if exactly 4 ranks exist),
        // tiles are exchanged between ranks to simulate dynamic load balancing.
        if (step == 5 && size == 4) {
            if (rank == 0 || rank == 1 || rank == 2) {
                // For simplicity, send the tile at index 3 if it exists
                if (info.tiles.size() > 3) {
                    removeTileFromRank(info, owner, 3, 3);
                }
            } else if (rank == 3) {
                // Rank 3 expects to receive 3 tiles from ranks 0, 1, and 2
                for (int source = 0; source < 3; source++) {
                    addTileToRank(info, owner, source, globalTileCols, interior_nx, interior_ny, guard);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }

        // 4) Save Simulation Data:
        if (step % save_frequency == 0) {
            std::string filename = "Simulation/Fields/fields_rank_" + std::to_string(rank) + "_step_" + std::to_string(step) + ".h5";
            saveRankData(info, guard, filename);
        }
        
    } // End of simulation loop

    // Final Output: Print tile information after simulation completes
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p = 0; p < size; p++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p) {
            cout << "\n===== FINAL OUTPUT FOR RANK " << rank
                 << " (Rank coords: " << rankRow << "," << rankCol
                 << ") after " << total_steps << " steps =====\n";
            for (int i = 0; i < info.tiles.size(); i++) {
                Tile &tile = info.tiles[i];
                cout << "Tile GID=" << tile.info.globalID
                     << " (Row=" << tile.info.tileRow
                     << ", Col=" << tile.info.tileCol
                     << ", currentRank=" << tile.info.currentRank << ")\n";
            }
            cout.flush();
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "\nSimulation complete.\n";
    }
    MPI_Finalize();
    return 0;
}
