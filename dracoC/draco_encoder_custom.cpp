#include <string>
#include <fstream>
#include <iostream>
#include "draco/compression/encode.h"
#include "draco/mesh/triangle_soup_mesh_builder.h"
#include "draco/io/file_utils.h"
#include "draco/io/mesh_io.h"
#include "draco/io/file_writer_factory.h"
#include "draco/io/file_writer_interface.h"
#include "draco/io/stdio_file_writer.h"
#include <cstdlib>
#include <vector>
#include <sstream>
// https://stackoverflow.com/questions/62313136/how-to-include-the-draco-dec-library
// g++ dracoEncoding.cpp -I /groups/scicompsoft/home/ackermand/local/include/ -L /groups/scicompsoft/home/ackermand/local/lib/ -ldraco -o convertToQuantizedDracoMesh

// Function object that quantizes an input `std::array<float, 3>` vertex
// position to the specified number of bits.
//
// \tparam VertexCoord Unsigned integer type used for representing quantized
//     vertex coordinates.

using Vector3d = std::array<float,3>;//std::array<int64_t, 3>;
using VertexCoord = std::uint32_t;

struct Quantizer {
  // Constructs a quantizer.
  //
  // \param fragment_origin Minimum input vertex position to represent.
  // \param fragment_shape The inclusive maximum vertex position to represent
  //     is `fragment_origin + fragment_shape`.
  // \param input_origin The offset to add to input vertices before quantizing
  //     them within the `[fragment_origin, fragment_origin+fragment_shape]`
  //     range.
  // \param num_quantization_bits The number of bits to use for quantization.
  //     A value of `0` for coordinate `i` corresponds to `fragment_origin[i]`,
  //     while a value of `2**num_quantization_bits-1` corresponds to
  //     `fragment_origin[i]+fragment_shape[i]`.  Should be less than or equal
  //     to the number of bits in `VertexCoord`.
  Quantizer(const int* fragment_origin, const Vector3d& fragment_shape,
            const Vector3d& input_origin, int num_quantization_bits) {
    for (int i = 0; i < 3; ++i) {
      upper_bound[i] =
          static_cast<float>(std::numeric_limits<VertexCoord>::max() >>
                             (sizeof(VertexCoord) * 8 - num_quantization_bits));
      scale[i] = upper_bound[i] / static_cast<float>(fragment_shape[i]);
      // Add 0.5 to round to nearest rather than round down.
      offset[i] = input_origin[i] - fragment_origin[i] + 0.5 / scale[i];
    }
  }

  // Maps an input vertex position `v_pos`.
 // std::array<VertexCoord, 3> operator()(const std::array<float, 3>& v_pos) {
  std::array<VertexCoord, 3> operator()(std::vector<float> v_pos) {
    std::array<VertexCoord, 3> output;
    for (int i = 0; i < 3; ++i) {
      output[i] = static_cast<VertexCoord>(std::min(
          upper_bound[i], std::max(0.0f, (v_pos.at(i) + offset[i]) * scale[i])));
    }
    return output;
  }

  std::array<float, 3> offset;
  std::array<float, 3> scale;
  std::array<float, 3> upper_bound;
};

struct Mesh{
  std::vector<std::vector<float>> vertices;
  std::vector<std::vector<int>> faces;
};

Mesh ReadObjFile(std::string & input_file){
  std::ifstream myfile (input_file);
  std::string line;

  std::vector<std::vector<float>> vertices;
  std::vector<std::vector<int>> faces;

  std::string v;
  float x,y,z;
  while(!myfile.eof())
  {
      getline (myfile,line);
      if (line[0] == 'v')
      {
          std::vector<float> vertex;
          std::istringstream iss( line );
          iss>>v>>x>>y>>z;
          vertex.push_back(x);
          vertex.push_back(y);
          vertex.push_back(z);

          vertices.push_back(vertex);
      }
      else if(line[0] == 'f'){
          std::vector<int> face;
          std::istringstream iss( line );
          iss>>v>>x>>y>>z;
          face.push_back(x);
          face.push_back(y);
          face.push_back(z);

          faces.push_back(face);
      }
  }

  Mesh mesh = {vertices,faces};
  return mesh;
}

void ConvertToDracoMeshWithPreQuantization(std::string & input_file, std::string & output_file, int * fragment_origin){
  Mesh mesh = ReadObjFile(input_file);
  int num_quantization_bits = 10;

  Vector3d fragment_shape = {1,1,1};
  Vector3d input_origin = {0,0,0};
  Quantizer quantizer(fragment_origin, fragment_shape,
                                     input_origin, num_quantization_bits);

  draco::TriangleSoupMeshBuilder mb;
  using VertexCoord = std::uint32_t;

  int num_triangles = mesh.faces.size();
  if(num_triangles>0){
    mb.Start(num_triangles);
    int position_att_id = mb.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_UINT32);
    int face_index = 0;
    for (std::vector<int> face : mesh.faces){

        std::array<std::array<VertexCoord, 3>, 3> points;
        for (int i = 0; i < 3; ++i) {
          int face_index = face.at(i) -1;
          points[i] = quantizer(mesh.vertices.at( face_index ));
        }
      
        mb.SetAttributeValuesForFace(position_att_id, draco::FaceIndex(face_index),
                                       &points[0], &points[1], &points[2]);
        face_index++;
    }

    std::unique_ptr<draco::Mesh> draco_mesh = mb.Finalize();  
    draco::EncoderBuffer eb;
    draco::Encoder encoder;
    encoder.SetEncodingMethod(draco::MESH_EDGEBREAKER_ENCODING);
    encoder.SetAttributePredictionScheme(draco::GeometryAttribute::POSITION,
                                         draco::MESH_PREDICTION_PARALLELOGRAM);
    auto draco_status = encoder.EncodeMeshToBuffer(*draco_mesh, &eb);
       
    draco::FileWriterFactory::RegisterWriter(draco::StdioFileWriter::Open);
    draco::WriteBufferToFile(eb.data(), eb.size(), output_file);
  }

}

int main(int argc, char ** argv){

  std::string input_file = argv[1];
  std::string output_file = argv[2];
  int fragment_origin[3]= {std::atoi(argv[3]),std::atoi(argv[4]),std::atoi(argv[5])};
  ConvertToDracoMeshWithPreQuantization(input_file, output_file, fragment_origin);
    
  return 0;
}