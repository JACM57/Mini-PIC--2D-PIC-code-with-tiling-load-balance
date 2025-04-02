#include "Auxiliar_functions.h"
#include <cmath>
#include <vector>
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <cstdio>

using namespace std;

// Compute a near-square decomposition (R rows x C columns) for 'size' processes
void findBestGrid(int size, int &R, int &C) {
    R = static_cast<int>(sqrt((double)size));
    while (R > 1 && (size % R != 0)) {
        R--;
    }
    C = size / R;
}

// Compute a near-square decomposition for 'numTiles' per rank
void findBestTileGrid(int numTiles, int &tileRows, int &tileCols) {
    tileRows = static_cast<int>(sqrt((double)numTiles));
    while (tileRows > 1 && (numTiles % tileRows != 0)) {
        tileRows--;
    }
    tileCols = numTiles / tileRows;
}

// Compute the global row for a tile given the rank's row and the tile's local row index
int tileGlobalRow(int rankRow, int tileRows, int tRow) {
    return rankRow * tileRows + tRow;
}

// Compute the global column for a tile given the rank's col and the tile's local col index
int tileGlobalCol(int rankCol, int tileCols, int tCol) {
    return rankCol * tileCols + tCol;
}

// Convert (globalRow, globalCol) to a unique tile ID in row-major order
int getGlobalID(int globalRow, int globalCol, int globalTileCols) {
    return globalRow * globalTileCols + globalCol;
}

// Recover (globalRow, globalCol) from the global tile ID
void getGlobalRowCol(int gID, int globalTileCols, int &row, int &col) {
    row = gID / globalTileCols;
    col = gID % globalTileCols;
}

// Return the globalID of a neighbor tile in direction d with 2D periodic wrapping
int getNeighborGID(int myGID, int d, int globalTileRows, int globalTileCols) {
    int row, col;
    getGlobalRowCol(myGID, globalTileCols, row, col);
    int newRow = row + dRow[d];
    int newCol = col + dCol[d];
    if (newRow < 0) newRow = globalTileRows - 1;
    else if (newRow >= globalTileRows) newRow = 0;
    if (newCol < 0) newCol = globalTileCols - 1;
    else if (newCol >= globalTileCols) newCol = 0;
    return getGlobalID(newRow, newCol, globalTileCols);
}

// Compute a unique MPI tag based on the tile's global ID and communication direction
int computeTag(int tileGID, int direction) {
    return tileGID * 8 + direction;
}

// packSendBuffer: Extracts the guard region from a tile in a specified direction (d)
vector<Grid> packSendBuffer(const Tile &tile, int d, int guard) {
    int totalX = tile.info.nx + 2 * guard;
    int totalY = tile.info.ny + 2 * guard;
    vector<Grid> sendBuffer;
    int sendSize = 0;
    
    // For each direction, the appropriate guard cells are copied into a send buffer
    if (d == 0) { // Left
        sendSize = tile.info.ny * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < tile.info.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 1) { // Right
        sendSize = tile.info.ny * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < tile.info.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + (tile.info.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 2) { // Up
        sendSize = tile.info.nx * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.info.nx; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * tile.info.nx + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 3) { // Down
        sendSize = tile.info.nx * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.info.nx; col++) {
                int srcRow = guard + (tile.info.ny - guard) + row;
                int srcCol = guard + col;
                sendBuffer[row * tile.info.nx + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 4) { // Top-left
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 5) { // Top-right
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + row;
                int srcCol = guard + (tile.info.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 6) { // Bottom-left
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + (tile.info.ny - guard) + row;
                int srcCol = guard + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    else if (d == 7) { // Bottom-right
        sendSize = guard * guard;
        sendBuffer.resize(sendSize);
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                int srcRow = guard + (tile.info.ny - guard) + row;
                int srcCol = guard + (tile.info.nx - guard) + col;
                sendBuffer[row * guard + col] = tile.grid[srcRow * totalX + srcCol];
            }
        }
    }
    return sendBuffer;
}

// updateGuardRegion: Updates the tile's guard region in direction d with received data from rbuf
void updateGuardRegion(Tile &tile, int d, int guard, const vector<Grid> &rbuf) {
    int totalX = tile.info.nx + 2 * guard;
    int totalY = tile.info.ny + 2 * guard;
    if (d == 0) { // Left
        for (int row = 0; row < tile.info.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int dstRow = guard + row;
                int dstCol = col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 1) { // Right
        for (int row = 0; row < tile.info.ny; row++) {
            for (int col = 0; col < guard; col++) {
                int dstRow = guard + row;
                int dstCol = totalX - guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 2) { // Up
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.info.nx; col++) {
                int dstRow = row;
                int dstCol = guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * tile.info.nx + col];
            }
        }
    }
    else if (d == 3) { // Down
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < tile.info.nx; col++) {
                int dstRow = totalY - guard + row;
                int dstCol = guard + col;
                tile.grid[dstRow * totalX + dstCol] = rbuf[row * tile.info.nx + col];
            }
        }
    }
    else if (d == 4) { // Top-left
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[row * totalX + col] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 5) { // Top-right
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[row * totalX + (totalX - guard + col)] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 6) { // Bottom-left
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[(totalY - guard + row) * totalX + col] = rbuf[row * guard + col];
            }
        }
    }
    else if (d == 7) { // Bottom-right
        for (int row = 0; row < guard; row++) {
            for (int col = 0; col < guard; col++) {
                tile.grid[(totalY - guard + row) * totalX + (totalX - guard + col)] = rbuf[row * guard + col];
            }
        }
    }
}

// removeTileFromRank: Remove a tile (send it to the destination rank) and remove it from the local RankInfo
void removeTileFromRank(RankInfo &info, vector<int> &owner, int tileIndex, int dest) {
    Tile movingTile = info.tiles[tileIndex];
    MPI_Send(&movingTile.info.globalID, 1, MPI_INT, dest, 999, MPI_COMM_WORLD);
    int nGrid = movingTile.grid.size();
    MPI_Send(&movingTile.grid[0], nGrid * sizeof(Grid), MPI_BYTE, dest, 1000, MPI_COMM_WORLD);
    owner[movingTile.info.globalID] = dest;
    info.tiles.erase(info.tiles.begin() + tileIndex);
    cout << "Rank " << info.rank << " sent Tile GID=" << movingTile.info.globalID << " to Rank " << dest << endl << flush;
}

// addTileToRank: Add a tile (receive it from a source rank) and add it to the local RankInfo
void addTileToRank(RankInfo &info, vector<int> &owner, int source, int globalTileCols, int interior_nx, int interior_ny, int guard) {
    int inGID;
    MPI_Recv(&inGID, 1, MPI_INT, source, 999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    Tile newTile;
    newTile.info.globalID = inGID;
    newTile.info.nx = interior_nx;
    newTile.info.ny = interior_ny;
    newTile.info.currentRank = info.rank;
    int totalX = interior_nx + 2 * guard;
    int totalY = interior_ny + 2 * guard;
    newTile.grid.resize(totalX * totalY);
    MPI_Recv(&newTile.grid[0], newTile.grid.size() * sizeof(Grid), MPI_BYTE, source, 1000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int r, c;
    getGlobalRowCol(inGID, globalTileCols, r, c);
    newTile.info.tileRow = r;
    newTile.info.tileCol = c;
    owner[inGID] = info.rank;
    info.tiles.push_back(newTile);
    cout << "Rank " << info.rank << " received Tile GID=" << inGID << " from Rank " << source << endl << flush;
}

// Delete folder for new data saving
void deleteFolder(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (!dir) return;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;

        std::string fullPath = path + "/" + name;

        if (entry->d_type == DT_DIR) {
            deleteFolder(fullPath);
            rmdir(fullPath.c_str());
        } else {
            unlink(fullPath.c_str());
        }
    }
    closedir(dir);
    rmdir(path.c_str());
}

double heaviside(double x) {
    return (x >= 0) ? 1.0 : 0.0;
}