#pragma once

#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>
//https://stackoverflow.com/a/217605
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>
#include <random>
#include <memory>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <opencv2/highgui/highgui.hpp>

// //loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>


//https://channel9.msdn.com/Events/GoingNative/2013/rand-Considered-Harmful
// https://kristerw.blogspot.com/2017/05/seeding-stdmt19937-random-number-engine.html
class RandGenerator{
public:
    RandGenerator(): 
        // m_gen((std::random_device())()) //https://stackoverflow.com/a/29580889
        m_gen(0) //start with a defined seed
        {

    }
    //returns a random float in the range [a,b], inclusive
    float rand_float(float a, float b) {
        std::uniform_real_distribution<float> distribution(a,b); 
        return distribution(m_gen);
    }

    //returns a random float with a normal distribution with mean and stddev
    float rand_normal_float(float mean, float stddev) {
        std::normal_distribution<float> distribution(mean, stddev); 
        return distribution(m_gen);
    }

    //returns a random int in the range between [a,b] inclusive
    int rand_int(int a, int b) {
        std::uniform_int_distribution<int> distribution(a,b); 
        return distribution(m_gen);
    }

    //return a randomly bool with a probability of true of prob_true
    bool rand_bool(const float prob_true){
        std::bernoulli_distribution distribution(prob_true);
        return distribution(m_gen);
    }

    std::mt19937& generator(){
        return m_gen;
    }


private:
    std::mt19937 m_gen;
};



typedef std::vector<float> row_type_f;
typedef std::vector<row_type_f> matrix_type_f;

typedef std::vector<double> row_type_d;
typedef std::vector<row_type_f> matrix_type_d;

typedef std::vector<int> row_type_i;
typedef std::vector<row_type_i> matrix_type_i;

typedef std::vector<bool> row_type_b;
typedef std::vector<row_type_b> matrix_type_b;

namespace easy_pbr{
namespace utils{

// Converts degrees to radians.
inline float degrees2radians(float angle_degrees){
    return  (angle_degrees * M_PI / 180.0);
} 

// Converts radians to degrees.
inline float radians2degrees(float angle_radians){
    return (angle_radians * 180.0 / M_PI);
} 
//clamp a value between a min and a max
template <class T>
inline T clamp(const T val, const T min, const T max){
    return std::min(std::max(val, min),max);
}

//Best answer of https://stackoverflow.com/questions/5731863/mapping-a-numeric-range-onto-another
inline float map(const float input, const float input_start,const float input_end, const float output_start, const float output_end) {
    //we clamp the input between the start and the end 
    float input_clamped=clamp(input, input_start, input_end);
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start);
}

//smoothstep like in glsl https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/smoothstep.xhtml
inline float smoothstep(const float edge0, const float edge1, const float x){
    CHECK(edge0 < edge1) << "The GLSL code for smoothstep only allows a transition from a lower number to a bigger one. Didn't have bother to modify this.";
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0 - 2.0 * t);
}

//FROM Fast and Funky 1D Nonlinear Transformations:  https://www.youtube.com/watch?v=mr5xkf6zSzk
//Cubic (3d degree) Bezier through A,B,C,D where A(start) and D(end) are assumed to be 1
inline float normalized_bezier(float B, float C, float t){
    CHECK(t<=1.0 && t>=0.0) << "t must be in range [0,1]";

    float s = 1.0f - t;
    float t2 = t*t;
    float s2 = s*s;
    float t3 = t2*2;
    return (3.0*B*s2*t) + (3.0*C*s*t2) + t3;
}


//Adapted from https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
//given a certain value from a range between vmin and vmax we output the corresponding jet colormap color
inline std::vector<float> jet_color(double v, double vmin, double vmax) {
    std::vector<float> c = {1.0, 1.0, 1.0}; // white
    double dv;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c[0] = 0;
        c[1] = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c[0] = 0;
        c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c[0] = 4 * (v - vmin - 0.5 * dv) / dv;
        c[2] = 0;
    } else {
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c[2] = 0;
    }

    return (c);
}

//https://stackoverflow.com/questions/7276826/c-format-number-with-commas
template<class T>
std::string format_with_commas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}



inline Eigen::Vector3d random_color(RandGenerator& rand_gen) {
    Eigen::Vector3d color;
    color(0) = rand_gen.rand_float(0.0, 1.0);
    color(1) = rand_gen.rand_float(0.0, 1.0);
    color(2) = rand_gen.rand_float(0.0, 1.0);
    return color;
}

inline Eigen::Vector3d random_color(std::shared_ptr<RandGenerator> rand_gen) {
    Eigen::Vector3d color;
    color(0) = rand_gen->rand_float(0.0, 1.0);
    color(1) = rand_gen->rand_float(0.0, 1.0);
    color(2) = rand_gen->rand_float(0.0, 1.0);
    return color;
}


inline void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

inline void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

inline void EigenToFile(Eigen::MatrixXd& src, std::string pathAndName)
{
      std::ofstream fichier(pathAndName);
      if(fichier.is_open())  // si l'ouverture a réussi
      {
        // instructions
        fichier << src << "\n";
        fichier.close();  // on referme le fichier
      }
      else  // sinon
      {
        std::cerr << "Erreur à l'ouverture !" << std::endl;
      }
 }


inline std::string file_to_string (const std::string &filename){
    std::ifstream t(filename);
    if (!t.is_open()){
        LOG(FATAL) << "Cannot open file " << filename;
    }
    return std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
}


//Needed because % is not actually modulo in c++ and it may yield unexpected valued for negative numbers
//https://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c
inline int mod(int k, int n) {
    return ((k %= n) < 0) ? k+n : k;
}

// //for dynamic eigen matrices, where you don't need the eigen aligned alocator
// template<class T>
// inline Eigen::Matrix<T, -1, 1> vec2eigen( const std::vector< Eigen::Matrix<T, -1, 1> >& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     if(std_vec.size()==0){
//         Eigen::Matrix<T, -1, 1> eigen_vec;
//         return eigen_vec;
//     }

//     int dim=std_vec[0].size();
//     Eigen::Matrix<T, -1, 1> eigen_vec(std_vec.size(),dim);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

// //for row being 1x3 matrices
// template<typename T, int Rows, int Cols> //the type of the matrix (float, int, bool, etc)
// // template <typename R>  //the number of columns in each row (usually it's 3)
// inline Eigen::Matrix<T, -1, 1> vec2eigen( const std::vector< Eigen::Matrix<T, Rows, Cols>, Eigen::aligned_allocator<Eigen::Matrix<T, Rows, Cols>>   >& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     if(std_vec.size()==0){
//         //return empty eigen mat
//         Eigen::Matrix<T, -1, 1> eigen_vec;
//         return eigen_vec;
//     }

//     int dim=std_vec[0].size();
//     Eigen::Matrix<T, -1, 1> eigen_vec(std_vec.size(),dim);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

// template<typename T, int Rows, int Cols> //the type of the matrix (float, int, bool, etc)
// template <typename R>  //the number of columns in each row (usually it's 3)
// template <typename Derived>
// inline Eigen::PlainObjectBase<Derived> vec2eigen( const std::vector< Eigen::EigenBase<Derived> , Eigen::aligned_allocator<Eigen::EigenBase<Derived>>   >& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     // if(std_vec.size()==0){
//     //     //return empty eigen mat
//     //     Eigen::Matrix<T, -1, 1> eigen_vec;
//     //     return eigen_vec;
//     // }

//     // int dim=std_vec[0].size();
//     // Eigen::Matrix<T, -1, 1> eigen_vec(std_vec.size(),dim);
//     // for (size_t i = 0; i < std_vec.size(); ++i) {
//     //     eigen_vec.row(i)=std_vec[i];
//     // }
//     // return eigen_vec;

//     const int dim=std_vec[0].size();
//     Eigen::Matrix<typename Derived::Scalar, std_vec.size() , dim> eigen_vec;

//     return eigen_vec;
// }


// template <typename Derived>
// inline Eigen::PlainObjectBase<Derived> vec2eigen( const std::vector< Eigen::MatrixBase<Derived>  >& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     // if(std_vec.size()==0){
//     //     //return empty eigen mat
//     //     Eigen::Matrix<T, -1, 1> eigen_vec;
//     //     return eigen_vec;
//     // }

//     // int dim=std_vec[0].size();
//     // Eigen::Matrix<T, -1, 1> eigen_vec(std_vec.size(),dim);
//     // for (size_t i = 0; i < std_vec.size(); ++i) {
//     //     eigen_vec.row(i)=std_vec[i];
//     // }
//     // return eigen_vec;

//     const int dim=std_vec[0].size();
//     Eigen::Matrix<typename Derived::Scalar, std_vec.size() , dim> eigen_vec;

//     return eigen_vec;
// }

//when using dyanmic vector we don't need an eigen alocator
template<class T> 
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector<  Eigen::Matrix<T, Eigen::Dynamic, 1>>& std_vec )
{
    if(std_vec.size()==0){
        //return empty eigen mat
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
        return eigen_vec;
    }

    const int dim=std_vec[0].size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),dim);
    for (size_t i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;

}

// // if for any reason you have a dynamic vector and still added the Eigen aligned allcoator (as c++ 17 seems to automatically do...)
// template<class T> 
// Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector<  Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::aligned_allocator< Eigen::Matrix<T, Eigen::Dynamic, 1>>    >& std_vec )
// {
//     if(std_vec.size()==0){
//         //return empty eigen mat
//         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
//         return eigen_vec;
//     }

//     const int dim=std_vec[0].size();
//     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),dim);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;

// }


//for the case of using fixed sized vector like Vector3f instead of a dynamic VectorXf, we need an aligned allocator
template<class T, int rows > 
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector<  Eigen::Matrix<T, rows, 1>, Eigen::aligned_allocator< Eigen::Matrix<T, rows, 1>>   >& std_vec )
{
    if(std_vec.size()==0){
        //return empty eigen mat
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
        return eigen_vec;
    }

    const int dim=std_vec[0].size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),dim);
    for (size_t i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;

}


// //for the case of using fixed sized vector like Vector3f instead of a dynamic VectorXf, we need an aligned allocator
// template<class T, int rows > 
// Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector<  Eigen::Matrix<T, rows, 1>  >& std_vec )
// {
//     if(std_vec.size()==0){
//         //return empty eigen mat
//         Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
//         return eigen_vec;
//     }

//     const int dim=std_vec[0].size();
//     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),dim);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;

// }




//for the case of using fixed sized vector like Vector3f instead of a dynamic VectorXf, we need an aligned allocator
template<class T> 
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vec2eigen( const std::vector< T >& std_vec )
{
    if(std_vec.size()==0){
        //return empty eigen mat
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec;
        return eigen_vec;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_vec(std_vec.size(),1);
    for (size_t i = 0; i < std_vec.size(); ++i) {
        eigen_vec(i)=std_vec[i];
    }
    return eigen_vec;

}














// inline Eigen::MatrixXd vec2eigen( const std::vector<Eigen::VectorXd>& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     if(std_vec.size()==0){
//         Eigen::MatrixXd eigen_vec;
//         return eigen_vec;
//     }

//     int dim=std_vec[0].size();
//     Eigen::MatrixXd eigen_vec(std_vec.size(),dim);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

// inline Eigen::MatrixXi vec2eigen( const std::vector<Eigen::VectorXi>& std_vec, bool debug=false){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     if(std_vec.size()==0){
//         Eigen::MatrixXi eigen_vec;
//         return eigen_vec;
//     }

//     int dim=std_vec[0].size();
//     Eigen::MatrixXi eigen_vec(std_vec.size(),dim);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

// inline Eigen::MatrixXi vec2eigen( const std::vector<bool>& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd
//     Eigen::MatrixXi eigen_vec (std_vec.size(),1);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec(i)= std_vec[i] ? 1 : 0;
//     }
//     return eigen_vec;
// }

// inline Eigen::MatrixXd vec2eigen( const std::vector<double>& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd
//     Eigen::MatrixXd eigen_vec(std_vec.size(),1);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

// inline Eigen::MatrixXd vec2eigen( const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>  >& std_vec ){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     if(std_vec.size()==0){
//         Eigen::MatrixXd eigen_vec;
//         return eigen_vec;
//     }

//     Eigen::MatrixXd eigen_vec(std_vec.size(),3);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

// inline Eigen::MatrixXi vec2eigen( const std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>    >& std_vec){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     if(std_vec.size()==0){
//         Eigen::MatrixXi eigen_vec;
//         return eigen_vec;
//     }

//     Eigen::MatrixXi eigen_vec(std_vec.size(),3);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

// inline Eigen::MatrixXi vec2eigen( const std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> >& std_vec){
//     //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

//     if(std_vec.size()==0){
//         Eigen::MatrixXi eigen_vec;
//         return eigen_vec;
//     }

//     Eigen::MatrixXi eigen_vec(std_vec.size(),2);
//     for (size_t i = 0; i < std_vec.size(); ++i) {
//         eigen_vec.row(i)=std_vec[i];
//     }
//     return eigen_vec;
// }

//filters the rows of an eigen matrix and returns only those for which the mask is equal to the keep
template <class T>
inline T filter_impl(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks ){
    if(eigen_mat.rows()!=(int)mask.size() || eigen_mat.size()==0 ){
        LOG_IF_S(WARNING, do_checks) << "filter: Eigen matrix and mask don't have the same size: " << eigen_mat.rows() << " and " << mask.size() ;
        return eigen_mat;
    }

    int nr_elem_to_keep=std::count(mask.begin(), mask.end(), keep);
    T new_eigen_mat( nr_elem_to_keep, eigen_mat.cols() );

    indirection.resize(eigen_mat.rows(),-1);
    inverse_indirection.resize(nr_elem_to_keep, -1);


    int insert_idx=0;
    for (int i = 0; i < eigen_mat.rows(); ++i) {
        if(mask[i]==keep){
            new_eigen_mat.row(insert_idx)=eigen_mat.row(i);
            indirection[i]=insert_idx;
            inverse_indirection[insert_idx]=i;
            insert_idx++;
        }
    }

    return new_eigen_mat;

}

template <class T>
inline T filter( const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> indirection;
    std::vector<int> inverse_indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_return_indirection(std::vector<int>&indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> inverse_indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_return_inverse_indirection(std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true ){

    std::vector<int> indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_return_both_indirection(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true ){

    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

//gets rid of the faces that are redirected to a -1 or edges that are also indirected into a -1
inline Eigen::MatrixXi filter_apply_indirection(const std::vector<int>&indirection, const Eigen::MatrixXi& eigen_mat ){

    if(!eigen_mat.size()){  //it's empty
        return eigen_mat;
    }

    if(eigen_mat.maxCoeff() > (int)indirection.size()){
        LOG(FATAL) << "filter apply_indirection: eigen_mat is indexing indirection at a higher position than allowed" << eigen_mat.maxCoeff() << " " << indirection.size();
    }

    std::vector<Eigen::VectorXi> new_eigen_mat_vec;

    for (int i = 0; i < eigen_mat.rows(); ++i) {

        Eigen::VectorXi row= eigen_mat.row(i);
        bool should_keep=true;
        for (int j = 0; j < row.size(); ++j) {
            if (indirection[row(j)]==-1){
                //it points to a an already removed point so we will not keep it
                should_keep=false;
            }else{
                //it point to a valid vertex so we change the idx so that it point to that one
                row(j) = indirection[row(j)];
            }
        }

        if(should_keep){
            new_eigen_mat_vec.push_back(row);
        }

    }

    return vec2eigen(new_eigen_mat_vec);

}

//gets rid of the faces that are redirected to a -1 or edges that are also indirected into a -1 AND also returns a mask (size eigen_mat x 1) with value of TRUE for those which were keps
inline Eigen::MatrixXi filter_apply_indirection_return_mask(std::vector<bool>& mask_kept, const std::vector<int>&indirection, const Eigen::MatrixXi& eigen_mat ){

    if(!eigen_mat.size()){  //it's empty
        return eigen_mat;
    }

    if(eigen_mat.maxCoeff() > (int)indirection.size()){
        LOG(FATAL) << "filter apply_indirection: eigen_mat is indexing indirection at a higher position than allowed" << eigen_mat.maxCoeff() << " " << indirection.size();
    }

    std::vector<Eigen::VectorXi> new_eigen_mat_vec;
    mask_kept.resize(eigen_mat.rows(),false);

    for (int i = 0; i < eigen_mat.rows(); ++i) {
        // LOG_IF_S(INFO,debug) << "getting row " << i;
        Eigen::VectorXi row= eigen_mat.row(i);
        // LOG_IF_S(INFO,debug) << "row is" << row;
        bool should_keep=true;
        for (int j = 0; j < row.size(); ++j) {
            // LOG_IF_S(INFO,debug) << "value at column " << j << " is " << row(j);
            // LOG_IF_S(INFO,debug) << "indirection has size " << indirection.size();
            // LOG_IF_S(INFO,debug) << "value of indirection at is  " <<  indirection[row(j)];
            if (indirection[row(j)]==-1){
                //it points to a an already removed point so we will not keep it
                should_keep=false;
            }else{
                //it point to a valid vertex so we change the idx so that it point to that one
                row(j) = indirection[row(j)];
            }
        }

        if(should_keep){
            // LOG_IF_S(INFO,debug) << "pushing new row " <<  row;
            new_eigen_mat_vec.push_back(row);
            // LOG_IF_S(INFO,debug) << "setting face " << i << " to kept";
            mask_kept[i]=true;
        }

    }

    // LOG_IF_S(INFO,debug) << "finished, doing a vec2eigen ";
    return vec2eigen(new_eigen_mat_vec);

}



//filters that does not actually remove the points, but just sets them to zero
template <class T>
inline T filter_set_zero_impl(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks ){
    if(eigen_mat.rows()!=(int)mask.size() || eigen_mat.size()==0 ){
        LOG_IF_S(WARNING, do_checks) << "filter: Eigen matrix and mask don't have the same size: " << eigen_mat.rows() << " and " << mask.size() ;
        return eigen_mat;
    }

    T new_eigen_mat( eigen_mat.rows(), eigen_mat.cols() );
    new_eigen_mat.setZero();

    indirection.resize(eigen_mat.rows(),-1);
    inverse_indirection.resize(eigen_mat.rows(), -1);


    for (int i = 0; i < eigen_mat.rows(); ++i) {
        if(mask[i]==keep){
            new_eigen_mat.row(i)=eigen_mat.row(i);
            indirection[i]=i;
            inverse_indirection[i]=i;
        }
    }

    return new_eigen_mat;

}

//sets the corresponding rows to zero instead of removing them
template <class T>
inline T filter_set_zero( const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> indirection;
    std::vector<int> inverse_indirection;
    return filter_set_zero_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}

template <class T>
inline T filter_set_zero_return_indirection(std::vector<int>&indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep, const bool do_checks=true){

    std::vector<int> inverse_indirection;
    return filter_set_zero_impl(indirection, inverse_indirection, eigen_mat, mask, keep, do_checks);

}



template <class T>
inline T concat(const T& mat_1, const T& mat_2){

    if(mat_1.cols()!=mat_2.cols() && mat_1.cols()!=0 && mat_2.cols()!=0){
        LOG(FATAL) << "concat: Eigen matrices don't have the same nr of columns: " << mat_1.cols() << " and " << mat_2.cols() ;
    }


    T mat_new(mat_1.rows() + mat_2.rows(), mat_1.cols());
    mat_new << mat_1, mat_2;
    return mat_new;
}

//swap a Vector2, if the input is not a vector2, swaps the first 2 elements
template <class T>
inline T swap(const T& vec){
    T swapped=vec;
    swapped(0)=vec(1);
    swapped(1)=vec(0);
    return swapped;
}

//https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

inline void set_default_color(Eigen::MatrixXd& C){
    C.col(0).setConstant(0.41);
    C.col(1).setConstant(0.58);
    C.col(2).setConstant(0.59);
}

inline Eigen::MatrixXd default_color(int nr_rows){
    Eigen::MatrixXd C(nr_rows,3);
    C.col(0).setConstant(0.41);
    C.col(1).setConstant(0.58);
    C.col(2).setConstant(0.59);
    return C;

}

inline bool XOR(bool a, bool b)
{
    return (a + b) % 2;
}

//convert an OpenCV type to a string value
inline std::string type2string(int type) {
    std::string r;

    unsigned char depth = type & CV_MAT_DEPTH_MASK;
    unsigned char chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}


//return the byteDepth of this cv mat. return one of the following  CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F
inline unsigned char type2byteDepth(int type) {
    unsigned char depth = type & CV_MAT_DEPTH_MASK;

    return depth;
}

// Removed them because now we support rgb texture in EasyGL and noy only 4 channel ones
// inline void create_alpha_mat(const cv::Mat& mat, cv::Mat_<cv::Vec4b>& dst){
//     std::vector<cv::Mat> matChannels;
//     cv::split(mat, matChannels);

//     cv::Mat alpha=cv::Mat(mat.rows,mat.cols, CV_8UC1);
//     alpha.setTo(cv::Scalar(255));
//     matChannels.push_back(alpha);

//     cv::merge(matChannels, dst);
// }

template <class T>
inline cv::Mat_<cv::Vec<T,4> > create_alpha_mat(const cv::Mat& mat){
    std::vector<cv::Mat> matChannels;
    cv::split(mat, matChannels);

    cv::Mat alpha=cv::Mat(mat.rows,mat.cols, matChannels[0].type());
    alpha.setTo(cv::Scalar(255));
    matChannels.push_back(alpha);

    cv::Mat_<cv::Vec<T,4> > out;
    cv::merge(matChannels, out);
    return out;
}




// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

//https://stackoverflow.com/a/37454181
inline std::vector<std::string> split(const std::string& str, const std::string& delim){
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do{
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

inline std::string lowercase(const std::string str){
    std::string new_str=str;
    std::transform(new_str.begin(), new_str.end(), new_str.begin(), ::tolower);
    return new_str;
}

inline std::string uppercase(const std::string str){
    std::string new_str=str;
    std::transform(new_str.begin(), new_str.end(), new_str.begin(), ::toupper);
    return new_str;
}





inline int next_power_of_two(int x) { // https://github.com/LWJGL/lwjgl3-wiki/wiki/2.6.1.-Ray-tracing-with-OpenGL-Compute-Shaders-(Part-I)
  x--;
  x |= x >> 1; // handle 2 bit numbers
  x |= x >> 2; // handle 4 bit numbers
  x |= x >> 4; // handle 8 bit numbers
  x |= x >> 8; // handle 16 bit numbers
  x |= x >> 16; // handle 32 bit numbers
  x++;
  return x;
}

//if the value gets bigger than the max it wraps back from 0, if its smaller than 0 it also wrap back from max
template <class T>
inline T wrap(const T val, const T max){
    T new_val = val;
  
    while(new_val >= max) new_val = (new_val - max);
    while(new_val < 0) new_val = (new_val + max);
  
    return new_val;
}

//from a 2D coordinate, get a linear one, with wraping
template <class T>
inline T idx_2D_to_1D_wrap(const T x, const T y, const T width, const T height){
  
    T x_wrap=wrap(x,width);
    T y_wrap=wrap(y,height);

    return y_wrap*width +x_wrap;
}

//from a 2D coordinate, get a linear one, without wrapping (thows error if accessing out of bounds)
template <class T>
inline T idx_2D_to_1D(const T x, const T y, const T width, const T height){
    CHECK(x<width) << "x is accessing out of bounds of the width. x is " << x << " width is " << width;
    CHECK(y<height) << "y is accessing out of bounds of the height. y is " << y << " height is " << height;

    T idx= y*width + x;
    CHECK(idx<width*height) << "idx will access out of bounds. idx is " << idx << " width*height is " << width*height;

    return idx;
}


//from a 1D coordinate, get a 2D one
template <class T>
inline Eigen::Vector2i idx_1D_to_2D(const T idx, const T width, const T height){

    T x=idx%width; 
    T y=idx/width; 

    CHECK(x<width) << "x is accessing out of bounds of the width. x is " << x << " width is " << width;
    CHECK(y<height) << "y is accessing out of bounds of the height. y is " << y << " height is " << height;

    Eigen::Vector2i pos;    
    pos.x()=x;
    pos.y()=y;

    return pos;
}

//from a 1D coordinate, get a 3D one assuming that things are stored in memory in order zyx where x is the fastest changing dimension and z is the slowest
//output is a idx in xyz with respect to the origin of the grid
inline Eigen::Vector3i idx_1D_to_3D(const int idx, const Eigen::Vector3i& grid_sizes){

    // int z = idx / (grid_sizes.x() *grid_sizes.y() ) ; //we do a full z channel when we sweapped both x and y
    // int y = z % grid_sizes.y();
    // int x = y % grid_sizes.x(); 

    //http://www.alecjacobson.com/weblog/?p=1425
    int x = idx % grid_sizes.x();
    int y = (idx - x)/grid_sizes.x() % grid_sizes.y();
    int z = ((idx - x)/grid_sizes.x()-y)/ grid_sizes.y();

    CHECK(x<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << x << " x span is " << grid_sizes.x() ;
    CHECK(y<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << y << " y span is " << grid_sizes.y();
    CHECK(z<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << z << " y span is " << grid_sizes.z();

    Eigen::Vector3i pos;    
    pos.x()=x;
    pos.y()=y;
    pos.z()=z;

    return pos;
}

//from a 3D coordinate, get a 1D one assuming that things are stored in memory in order zyx where x is the fastest changing dimension and z is the slowest
//input is a index in format xyz with respect to the origin of the grid 
inline int idx_3D_to_1D(const Eigen::Vector3i pos, const Eigen::Vector3i& grid_sizes){

    //http://www.alecjacobson.com/weblog/?p=1425
    int index = pos.x() + grid_sizes.x()*(pos.y()+grid_sizes.y()*pos.z());

    CHECK(pos.x()<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << pos.x() << " x span is " << grid_sizes.x() ;
    CHECK(pos.y()<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << pos.y() << " y span is " << grid_sizes.y();
    CHECK(pos.z()<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << pos.z() << " y span is " << grid_sizes.z();

    return index;
}

//from a 1D coordinate, get a 4D one assuming that things are stored in memory in order wzyx where x is the fastest changing dimension and w is the slowest
//output is a idx in xyzw with respect to the origin of the grid
inline Eigen::Vector4i idx_1D_to_4D(const int idx, const Eigen::Vector4i& grid_sizes){

    // int z = idx / (grid_sizes.x() *grid_sizes.y() ) ; //we do a full z channel when we sweapped both x and y
    // int y = z % grid_sizes.y();
    // int x = y % grid_sizes.x(); 

    //http://www.alecjacobson.com/weblog/?p=1425
    int x = idx % grid_sizes.x();
    int y = (idx - x)/grid_sizes.x() % grid_sizes.y();
    int z = ((idx - x)/grid_sizes.x()-y) % grid_sizes.z();
    int w = (((idx - x)/grid_sizes.x()-y)-z)/ grid_sizes.z(); 

    CHECK(x<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << x << " x span is " << grid_sizes.x() ;
    CHECK(y<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << y << " y span is " << grid_sizes.y();
    CHECK(z<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << z << " y span is " << grid_sizes.z();
    CHECK(w<grid_sizes.w()) << "w is accessing out of bounds of the w grid size. w is " << z << " w span is " << grid_sizes.w();

    Eigen::Vector4i pos;    
    pos.x()=x;
    pos.y()=y;
    pos.z()=z;
    pos.w()=w;

    return pos;
}

//from a 4D coordinate, get a 1D one assuming that things are stored in memory in order wzyx where x is the fastest changing dimension and w is the slowest
inline int idx_4D_to_1D(const Eigen::Vector4i pos, const Eigen::Vector4i& grid_sizes){

    //http://www.alecjacobson.com/weblog/?p=1425
    int index = pos.x() + grid_sizes.x()*(pos.y()+grid_sizes.y()*(pos.z() + grid_sizes.z()*pos.w())  );

    CHECK(pos.x()<grid_sizes.x()) << "x is accessing out of bounds of the x grid size. x is " << pos.x() << " x span is " << grid_sizes.x() ;
    CHECK(pos.y()<grid_sizes.y()) << "y is accessing out of bounds of the y grid size. y is " << pos.y() << " y span is " << grid_sizes.y();
    CHECK(pos.z()<grid_sizes.z()) << "z is accessing out of bounds of the z grid size. z is " << pos.z() << " y span is " << grid_sizes.z();
    CHECK(pos.w()<grid_sizes.w()) << "w is accessing out of bounds of the w grid size. w is " << pos.w() << " w span is " << grid_sizes.w();

    return index;
}

// To return char for a value. For example '2' 
// is returned for 2. 'A' is returned for 10. 'B' 
// for 11 
// Based on https://www.geeksforgeeks.org/convert-base-decimal-vice-versa/
inline char reVal(int num) { 
    if (num >= 0 && num <= 9) 
        return (char)(num + '0'); 
    else
        return (char)(num - 10 + 'A'); 
} 

//go from decimat base to any other base and returns the digit of that base as a std vector. Based on https://www.geeksforgeeks.org/convert-base-decimal-vice-versa/
//useful for creating uniform boxel grids in any dimension where vertices are defined as (0,0), (0,1), (1,0), (1,1) etc. Look at misc_utils/PermutoLatticePlotter for details
inline std::vector<int> convert_decimal_to_base(const int num, const int base) { 
    int index = 0;  // Initialize index of result 

    int val=num;
    std::vector<int> digits;
    // Convert input number is given base by repeatedly 
    // dividing it by base and taking remainder 
    while (val > 0) { 
        digits.push_back( val%base ); 
        val /= base; 
    } 
  
    // Reverse the result 
    std::reverse(digits.begin(), digits.end());
  
    return digits; 
} 

} //namespace utils
} //namespace er

