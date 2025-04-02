#ifndef AUXILIAR_FUNCTIONS_H
#define AUXILIAR_FUNCTIONS_H

#include <vector>
#include <mpi.h>
#include <cmath>

using std::vector;

// Communication arrays: displacements for neighbor communication and their opposites
static const int dRow[8] = { 0,  0, -1,  1, -1, -1,  1,  1 };
static const int dCol[8] = { -1, 1,  0,  0, -1,  1, -1,  1 };
static const int opposite[8] = { 1,  0,  3,  2,  7,  6,  5,  4 };

// Particle definition: holds particle properties (tag, charge, positions, momenta)
struct Particle {
    int tag;
    double q;
    double x, y;
    double px, py, pz;
};

// Grid structure: Stores combined electric (E) and magnetic (B) field components for each grid cell
struct Grid {
    double Ex, Ey, Ez;
    double Bx, By, Bz;
};

// TileInfo structure: Contains tile metadata (globalID, tile row/column, interior sizes, and the current owner rank)
struct TileInfo {
    int globalID;
    int tileRow, tileCol;
    int nx, ny;
    int currentRank;
};

// Tile structure: Combines tile metadata with its associated particles and grid cells (fields)
struct Tile {
    TileInfo info;
    vector<Particle> particles;
    vector<Grid> grid;
};

// RankInfo structure: Represents an MPI rank and lists the tiles currently owned by that rank
struct RankInfo {
    int rank, rankRow, rankCol;
    int tileRows, tileCols;
    vector<Tile> tiles;
};


// Auxiliar Functions

// Domain and tile decomposition
void findBestGrid(int size, int &R, int &C);
void findBestTileGrid(int numTiles, int &tileRows, int &tileCols);
int tileGlobalRow(int rankRow, int tileRows, int tRow);
int tileGlobalCol(int rankCol, int tileCols, int tCol);
int getGlobalID(int globalRow, int globalCol, int globalTileCols);
void getGlobalRowCol(int gID, int globalTileCols, int &row, int &col);

// Neighbor lookup and MPI tag computation
int getNeighborGID(int myGID, int d, int globalTileRows, int globalTileCols);
int computeTag(int tileGID, int direction);

// Communication routines for guard cells
vector<Grid> packSendBuffer(const Tile &tile, int d, int guard);
void updateGuardRegion(Tile &tile, int d, int guard, const vector<Grid> &rbuf);

// Dynamic tile migration helpers
void removeTileFromRank(RankInfo &info, vector<int> &owner, int tileIndex, int dest);
void addTileToRank(RankInfo &info, vector<int> &owner, int source, int globalTileCols, int interior_nx, int interior_ny, int guard);

// Filesystem utilities
void deleteFolder(const std::string& path);

// Help Function: Heaviside
double heaviside(double x);

#endif // AUXILIAR_FUNCTIONS_H
