#pragma once 

#include <algorithm>

#include <Eigen/Core>

//most of it is copied from https://github.com/yuki-koyama/tinycolormap/blob/master/include/tinycolormap.hpp

class ColormapMngr{
public:
    ColormapMngr();

    Eigen::MatrixXf magma_colormap();
    Eigen::Vector3f magma_color(const float x);
    Eigen::MatrixXf plasma_colormap();
    Eigen::Vector3f plasma_color(const float x);
    Eigen::MatrixXf viridis_colormap();
    Eigen::Vector3f viridis_color(const float x);


private:
    Eigen::MatrixXf m_magma_colormap;
    Eigen::MatrixXf m_plasma_colormap;
    Eigen::MatrixXf m_viridis_colormap;

};

