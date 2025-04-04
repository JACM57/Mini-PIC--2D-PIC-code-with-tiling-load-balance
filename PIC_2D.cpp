#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sys/stat.h>   
#include <sys/types.h>  
#include <dirent.h>     
#include <unistd.h>     
#include <cstdio>       
#include <cstring>     
#include "Auxiliar_functions.h"  // Contains helper functions and definitions
#include "Field_update.h"        // External routines for evolving fields
#include "HDF5_output.h"         // Function to save tile data to an HDF5 file

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine grid decomposition among ranks.
    int R, C;
    findBestGrid(size, R, C);
    int rankRow = rank / C;
    int rankCol = rank % C;

    // Each rank starts with 36 tiles (9x9 per rank)
    int numTiles = 36;
    int tileRows, tileCols;
    findBestTileGrid(numTiles, tileRows, tileCols);

    // Set up RankInfo for this rank.
    RankInfo info;
    info.rank = rank;
    info.rankRow = rankRow;
    info.rankCol = rankCol;
    info.tileRows = tileRows;
    info.tileCols = tileCols;
    info.tiles.resize(numTiles);

    // Global domain of tiles: (R*tileRows) x (C*tileCols)
    int globalTileRows = R * tileRows;
    int globalTileCols = C * tileCols;
    int totalGlobalTiles = globalTileRows * globalTileCols;

    // Create an ownership array: owner[gID] = rank that currently owns tile with globalID gID.
    vector<int> owner(totalGlobalTiles, -1);

    // Simulation parameters
    double box_x = 10.0, box_y = 10.0;   // physical dimensions
    // total grid cells in x and y
    //int nx = 300, ny = 300;      // For 4 cores
    //int nx = 450, ny = 180;      // For 5 cores
    int nx = 450, ny = 450;      // For 9 cores
    double dx = box_x / nx;
    double dy = box_y / ny;
    int guard = 2;                     // number of guard cells

    int interior_nx = nx / (C * tileCols);  // interior cells per tile in x
    int interior_ny = ny / (R * tileRows);  // interior cells per tile in y

    double sim_time = 500;
    double dt_courant = 1.0 / sqrt((1.0 / (dx*dx)) + (1.0 / (dy*dy)));
    double dt = 0.5 * dt_courant;
    int total_steps = static_cast<int>(sim_time / dt);
    int save_frequency = 25;

    // Tile initialization
    for (int t = 0; t < numTiles; t++) {
        Tile &tile = info.tiles[t];
        int tRow = t / tileCols; // local tile row
        int tCol = t % tileCols; // local tile col
        int gRow = tileGlobalRow(rankRow, tileRows, tRow); // global tile row
        int gCol = tileGlobalCol(rankCol, tileCols, tCol); // global tile col

        tile.info.globalID = getGlobalID(gRow, gCol, globalTileCols);
        tile.info.tileRow = gRow;
        tile.info.tileCol = gCol;
        tile.info.nx = interior_nx;
        tile.info.ny = interior_ny;
        tile.info.currentRank = rank;

        int totalX = interior_nx + 2 * guard;
        int totalY = interior_ny + 2 * guard;
        tile.grid.resize(totalX * totalY);

        // Global offsets (used to compute initial field values)
        // int global_y_offset = (globalTileRows - 1 - tile.info.tileRow) * interior_ny;
        int global_x_offset = tile.info.tileCol * interior_nx;
        int global_y_offset = tile.info.tileRow * interior_ny;

        // Prints for debugging
        //cout << "Rank " << rank << " Tile " << t << " Global ID: " << tile.info.globalID
        //     << " Global X Offset: " << global_x_offset << " Global Y Offset: " << global_y_offset << endl;

        double A = 0.1;
        double kx = 5 * 2 * M_PI / box_x;
        double ky = 5 * 2 * M_PI / box_y;
        double omega = 0;
        double x0 = 0.5;
        double xc = 3.5;
        double tau = 3.0;
        for (int j = 0; j < totalY; j++) {
            for (int i = 0; i < totalX; i++) {
                double x_Ex = (global_x_offset + i - guard + 0.5) * dx;
                double y_Ex = (global_y_offset + j - guard) * dy;
                double x_Ey = (global_x_offset + i - guard) * dx;
                double y_Ey = (global_y_offset + j - guard + 0.5) * dy;
                double x_Bz = (global_x_offset + i - guard + 0.5) * dx;
                double y_Bz = (global_y_offset + j - guard + 0.5) * dy;
                int idx = j * totalX + i;

                //tile.grid[idx].Ex = A * sin(ky * y_Ex);
                //tile.grid[idx].Ex = 0;
                //tile.grid[idx].Ex = (A / sqrt(2)) * sin(kx * x_Ex + ky * y_Ex);
                tile.grid[idx].Ex = 0;
                //tile.grid[idx].Ex =  A * sin(kx * y_Ex) * cos((y_Ex - xc)/tau * M_PI * 0.5) * cos((y_Ex - xc)/tau * M_PI * 0.5) * heaviside(1 - abs(y_Ex-xc)/tau);

                //tile.grid[idx].Ey = 0.0;
                //tile.grid[idx].Ey = A * sin(kx * x_Ey);
                //tile.grid[idx].Ey = - (A / sqrt(2)) * sin(kx * x_Ey + ky * y_Ey);
                tile.grid[idx].Ey = A * sin(kx * x_Ey) * cos((x_Ey - xc)/tau * M_PI * 0.5) * cos((x_Ey - xc)/tau * M_PI * 0.5) * heaviside(1 - abs(x_Ey-xc)/tau);
                //tile.grid[idx].Ey = 0.0;

                tile.grid[idx].Ez = 0.0;
                tile.grid[idx].Bx = 0.0;
                tile.grid[idx].By = 0.0;

                //tile.grid[idx].Bz = - A * sin(ky * y_Bz);
                //tile.grid[idx].Bz = A * sin(kx * x_Bz);
                //tile.grid[idx].Bz = -A * sin(kx * x_Bz + ky * y_Bz);
                tile.grid[idx].Bz = A * sin(kx * x_Bz) * cos((x_Bz - xc)/tau * M_PI * 0.5) * cos((x_Bz - xc)/tau * M_PI * 0.5) * heaviside(1 - abs(x_Bz-xc)/tau);
                //tile.grid[idx].Bz = - A * sin(kx * y_Bz) * cos((y_Bz - xc)/tau * M_PI * 0.5) * cos((y_Bz - xc)/tau * M_PI * 0.5) * heaviside(1 - abs(y_Bz-xc)/tau);
            }
        }
        owner[tile.info.globalID] = rank;
    }

    // Synchronize ownership info across all ranks.
    MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // Create/Clear folders for data saving
    if (rank == 0) {
        std::string sim_folder = "Simulation";
        std::string fields_folder = sim_folder + "/Fields";
    
        DIR* dir = opendir(fields_folder.c_str());
        if (dir) {
            closedir(dir);
            deleteFolder(fields_folder);
        }
    
        mkdir(sim_folder.c_str(), 0777);
        mkdir(fields_folder.c_str(), 0777);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Save initial state
    std::string filename = "Simulation/Fields/fields_rank_" + to_string(rank) + "_step_0.h5";
    saveRankData(info, guard, filename);

    // -------------------- Time-Stepping Simulation Loop -----------------------
    for (int step = 1; step < total_steps + 1; step++) {

        // --- Phase A: Update B Field (first half) ---
        for (int i = 0; i < (int)info.tiles.size(); i++) {
            Tile &tile = info.tiles[i];
            int totalX = tile.info.nx + 2 * guard;
            int totalY = tile.info.ny + 2 * guard;
            int totalCells = totalX * totalY;
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
            updateBhalf(Efull, BhalfOld, BhalfNew, interior_nx, interior_ny, guard, dt, dx, dy);
            for (int idx = 0; idx < totalCells; idx++) {
                tile.grid[idx].Bx = BhalfNew[idx].Bx;
                tile.grid[idx].By = BhalfNew[idx].By;
                tile.grid[idx].Bz = BhalfNew[idx].Bz;
            }
        }

        // --- Phase B: Guard-Cell Communication after B Update ---
        // Synchronize before starting communication
        MPI_Barrier(MPI_COMM_WORLD);
        {
            int localTileCount = info.tiles.size();
            int totalComm = localTileCount * 8;
            vector<MPI_Request> requests(totalComm * 2);
            int reqCount = 0;
            vector< vector< vector<Grid> > > recvBuffers(localTileCount, vector< vector<Grid> >(8));

            // Post receives first.
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    int nbrDirection = opposite[d];
                    int tag = computeTag(nbrGID, nbrDirection);
                    int bufferSize = (d < 2) ? interior_ny * guard :
                                     (d < 4) ? interior_nx * guard : guard * guard;
                    recvBuffers[i][d].resize(bufferSize);
                    MPI_Irecv(&recvBuffers[i][d][0], bufferSize * sizeof(Grid),
                              MPI_BYTE, nbrRank, tag, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            // A barrier here helps ensure all receives are posted
            MPI_Barrier(MPI_COMM_WORLD);
            // Post sends
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int tagSend = computeTag(gID, d);
                    vector<Grid> sendBuffer = packSendBuffer(tile, d, guard);
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    MPI_Isend(&sendBuffer[0], sendBuffer.size() * sizeof(Grid),
                              MPI_BYTE, nbrRank, tagSend, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Waitall(reqCount, &requests[0], MPI_STATUSES_IGNORE);
            // Update guard regions from received data
            for (int i = 0; i < localTileCount; i++) {
                for (int d = 0; d < 8; d++) {
                    updateGuardRegion(info.tiles[i], d, guard, recvBuffers[i][d]);
                }
            }
        } // End Phase B

        // --- Phase C: Update E Field ---
        for (int i = 0; i < (int)info.tiles.size(); i++) {
            Tile &tile = info.tiles[i];
            int totalX = tile.info.nx + 2 * guard;
            int totalY = tile.info.ny + 2 * guard;
            int totalCells = totalX * totalY;
            vector<GridE> Efull(totalCells);
            vector<GridB> Bhalf(totalCells);
            for (int idx = 0; idx < totalCells; idx++) {
                Efull[idx].Ex = tile.grid[idx].Ex;
                Efull[idx].Ey = tile.grid[idx].Ey;
                Efull[idx].Ez = tile.grid[idx].Ez;
                Bhalf[idx].Bx = tile.grid[idx].Bx;
                Bhalf[idx].By = tile.grid[idx].By;
                Bhalf[idx].Bz = tile.grid[idx].Bz;
            }
            updateEfull(Efull, Bhalf, interior_nx, interior_ny, guard, dt, dx, dy);
            for (int idx = 0; idx < totalCells; idx++) {
                tile.grid[idx].Ex = Efull[idx].Ex;
                tile.grid[idx].Ey = Efull[idx].Ey;
                tile.grid[idx].Ez = Efull[idx].Ez;
            }
        }

        // --- Phase D: Guard-Cell Communication after E Update ---
        // Again, ensure all ranks have finished updating E before posting guard exchanges
        MPI_Barrier(MPI_COMM_WORLD);
        {
            int localTileCount = info.tiles.size();
            int totalComm = localTileCount * 8;
            vector<MPI_Request> requests(totalComm * 2);
            int reqCount = 0;
            vector< vector< vector<Grid> > > recvBuffers(localTileCount, vector< vector<Grid> >(8));
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    int nbrDirection = opposite[d];
                    int tag = computeTag(nbrGID, nbrDirection);
                    int bufferSize = (d < 2) ? interior_ny * guard :
                                     (d < 4) ? interior_nx * guard : guard * guard;
                    recvBuffers[i][d].resize(bufferSize);
                    MPI_Irecv(&recvBuffers[i][d][0], bufferSize * sizeof(Grid),
                              MPI_BYTE, nbrRank, tag, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int tagSend = computeTag(gID, d);
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    vector<Grid> sendBuffer = packSendBuffer(tile, d, guard);
                    MPI_Isend(&sendBuffer[0], sendBuffer.size() * sizeof(Grid),
                              MPI_BYTE, nbrRank, tagSend, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Waitall(reqCount, &requests[0], MPI_STATUSES_IGNORE);
            for (int i = 0; i < localTileCount; i++) {
                for (int d = 0; d < 8; d++) {
                    updateGuardRegion(info.tiles[i], d, guard, recvBuffers[i][d]);
                }
            }
        } // End Phase D
        
        // --- Phase E: Update B Field (second half)---
        for (int i = 0; i < (int)info.tiles.size(); i++) {
            Tile &tile = info.tiles[i];
            int totalX = tile.info.nx + 2 * guard;
            int totalY = tile.info.ny + 2 * guard;
            int totalCells = totalX * totalY;
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
            updateBhalf(Efull, BhalfOld, BhalfNew, interior_nx, interior_ny, guard, dt, dx, dy);
            for (int idx = 0; idx < totalCells; idx++) {
                tile.grid[idx].Bx = BhalfNew[idx].Bx;
                tile.grid[idx].By = BhalfNew[idx].By;
                tile.grid[idx].Bz = BhalfNew[idx].Bz;
            }
        }
        
        // --- Phase F: Ghuard-Cell Communication after B (second alf) Update ---
        // Synchronize before starting communication.
        MPI_Barrier(MPI_COMM_WORLD);
        {
            int localTileCount = info.tiles.size();
            int totalComm = localTileCount * 8;
            vector<MPI_Request> requests(totalComm * 2);
            int reqCount = 0;
            vector< vector< vector<Grid> > > recvBuffers(localTileCount, vector< vector<Grid> >(8));

            // Post receives first.
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    int nbrDirection = opposite[d];
                    int tag = computeTag(nbrGID, nbrDirection);
                    int bufferSize = (d < 2) ? interior_ny * guard :
                                     (d < 4) ? interior_nx * guard : guard * guard;
                    recvBuffers[i][d].resize(bufferSize);
                    MPI_Irecv(&recvBuffers[i][d][0], bufferSize * sizeof(Grid),
                              MPI_BYTE, nbrRank, tag, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            // A barrier here helps ensure all receives are posted.
            MPI_Barrier(MPI_COMM_WORLD);
            // Post sends.
            for (int i = 0; i < localTileCount; i++) {
                Tile &tile = info.tiles[i];
                int gID = tile.info.globalID;
                for (int d = 0; d < 8; d++) {
                    int tagSend = computeTag(gID, d);
                    vector<Grid> sendBuffer = packSendBuffer(tile, d, guard);
                    int nbrGID = getNeighborGID(gID, d, globalTileRows, globalTileCols);
                    int nbrRank = owner[nbrGID];
                    MPI_Isend(&sendBuffer[0], sendBuffer.size() * sizeof(Grid),
                              MPI_BYTE, nbrRank, tagSend, MPI_COMM_WORLD, &requests[reqCount]);
                    reqCount++;
                }
            }
            MPI_Waitall(reqCount, &requests[0], MPI_STATUSES_IGNORE);
            // Update guard regions from received data.
            for (int i = 0; i < localTileCount; i++) {
                for (int d = 0; d < 8; d++) {
                    updateGuardRegion(info.tiles[i], d, guard, recvBuffers[i][d]);
                }
            }
        } // End Phase F

        // --- Phase G: Dynamic Tile Migration ---
        if (step == 5 && size == 9) {
            if (rank == 0 || rank == 1 || rank == 2) {
                if (info.tiles.size() > 3) {
                    removeTileFromRank(info, owner, 3, 3);
                }
            } else if (rank == 3) {
                for (int source = 0; source < 3; source++) {
                    addTileToRank(info, owner, source, globalTileCols, interior_nx, interior_ny, guard);
                }
            }
            // Synchronize and update owner info after migration
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &owner[0], totalGlobalTiles, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }

        // --- Phase H: Save Simulation Data ---
        if (step % save_frequency == 0) {
            std::string filename = "Simulation/Fields/fields_rank_" + to_string(rank) +
                                   "_step_" + to_string(step) + ".h5";
            saveRankData(info, guard, filename);
        }
    } // End simulation loop

    MPI_Barrier(MPI_COMM_WORLD);

    // Save data parameters in a .txt file for future plotting using File_Reader.py
    if (rank == 0) {
        std::ofstream paramFile("Simulation/Fields/params.txt");
        paramFile << "box_x=" << box_x << "\n";
        paramFile << "box_y=" << box_y << "\n";
        paramFile << "nx_global=" << nx << "\n";
        paramFile << "ny_global=" << ny << "\n";
        paramFile << "guard=" << guard << "\n";
        paramFile << "interior_nx=" << interior_nx << "\n";
        paramFile << "interior_ny=" << interior_ny << "\n";
        paramFile << "sim_time=" << sim_time << "\n";
        paramFile << "dt=" << dt << "\n";     
        paramFile << "total_steps=" << total_steps << "\n";
        paramFile.close();
    }

    // Final Output: Print tile info

    //for (int p = 0; p < size; p++) {
    //    MPI_Barrier(MPI_COMM_WORLD);
    //    if (rank == p) {
    //        cout << "\n===== FINAL OUTPUT FOR RANK " << rank
    //             << " (Rank coords: " << rankRow << "," << rankCol
    //             << ") after " << total_steps << " steps =====\n";
    //        for (int i = 0; i < (int)info.tiles.size(); i++) {
    //            cout << "Tile GID=" << info.tiles[i].info.globalID
    //                 << " (Row=" << info.tiles[i].info.tileRow
    //                 << ", Col=" << info.tiles[i].info.tileCol
    //                 << ", currentRank=" << info.tiles[i].info.currentRank << ")\n";
    //        }
    //        cout.flush();
    //    }
    //}
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "\nSimulation complete.\n";
    }
    MPI_Finalize();
    return 0;
}
