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
  Quantizer(const Vector3d& fragment_origin, const Vector3d& fragment_shape,
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

  Vector3d fragment_origin;
  Vector3d fragment_shape;
  Vector3d input_origin;
};

Mesh ReadObjFile(std::string &directory, Vector3d &fragment_shape, int *fragment_position){
  std::string filename=directory;
  for(int d=0; d<3; d++){
    filename+= std::to_string(fragment_position[d]);
  }
  filename+=".obj";
  std::ifstream myfile (filename);
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

  Vector3d input_origin = {0,0,0};
  Vector3d fragment_origin = {0,0,0};

  for (int d=0; d<3; d++){
    fragment_origin[d] = fragment_shape[d]*fragment_position[d];
  }

  Mesh mesh = {vertices,faces, fragment_origin, fragment_shape, input_origin};
  return mesh;
}
/*
// Combines multiple meshes into a single mesh, quantizes vertices according to
// the specified quantization options, and returns the Draco encoded result.
//
// \param meshes The list of meshes to combine.
// \param fragment_origin The minimum input vertex position to represent.
// \param fragment_shape The inclusive maximum vertex position to represent is
//     `fragment_origin+fragment_shape`.
// \param num_quantization_bits The number of bits to use for quantization.
std::string ConvertToDracoMeshWithPreQuantization(
    absl::Span<const InternalMesh> meshes, const Vector3d& fragment_origin,
    const Vector3d& fragment_shape, int num_quantization_bits) {
  draco::TriangleSoupMeshBuilder mb;
  size_t num_triangles = 0;
  for (const auto& mesh : meshes) {
    num_triangles += mesh.triangles.size();
  }
  mb.Start(num_triangles);
  // Draco always stores integer attributes as int32, so there is no point in
  // using a smaller integer type as the intermediate representation.
  using VertexCoord = std::uint32_t;
  int position_att_id =
      mb.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_UINT32);
  size_t tri_i = 0;
  for (const auto& mesh : meshes) {
    Quantizer<VertexCoord> quantizer(fragment_origin, fragment_shape,
                                     mesh.origin, num_quantization_bits);
    for (const auto& triangle : mesh.triangles) {
      std::array<std::array<VertexCoord, 3>, 3> points;
      for (int i = 0; i < 3; ++i) {
        points[i] = quantizer(mesh.vertex_positions[triangle[i]]);
      }
      mb.SetAttributeValuesForFace(position_att_id, draco::FaceIndex(tri_i),
                                   &points[0], &points[1], &points[2]);
      ++tri_i;
    }
  }
  if (tri_i == 0) {
    return std::string();
  }
  std::unique_ptr<draco::Mesh> draco_mesh = mb.Finalize();
  CHECK(draco_mesh);

  draco::EncoderBuffer eb;
  draco::Encoder encoder;
  encoder.SetEncodingMethod(draco::MESH_EDGEBREAKER_ENCODING);
  encoder.SetAttributePredictionScheme(draco::GeometryAttribute::POSITION,
                                       draco::MESH_PREDICTION_PARALLELOGRAM);
  auto draco_status = encoder.EncodeMeshToBuffer(*draco_mesh, &eb);
  CHECK(draco_status.ok()) << draco_status.error_msg_string();
  return std::string(eb.data(), eb.size());
}
*/

/* int testcase(){
  draco::TriangleSoupMeshBuilder mb;
  using VertexCoord = std::uint32_t;
  size_t tri = 1;
  mb.Start(tri);
  std::cout<<"part 0\n";
  int position_att_id =
  mb.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DT_UINT32);
  std::cout<<"part 1\n";
  std::cout<<"part 2\n";
  for (int tri_i=0; tri_i<tri; tri_i++){

      std::array<VertexCoord, 3> temp = {static_cast<VertexCoord>(0),static_cast<VertexCoord>(0),static_cast<VertexCoord>(0)};//static_cast<VertexCoord>(0),static_cast<VertexCoord>(0),static_cast<VertexCoord>(0)];
      std::array<VertexCoord, 3> temp1 = {static_cast<VertexCoord>(0),static_cast<VertexCoord>(10),static_cast<VertexCoord>(0)};//static_cast<VertexCoord>(0),static_cast<VertexCoord>(0),static_cast<VertexCoord>(0)];
      std::array<VertexCoord, 3> temp2 = {static_cast<VertexCoord>(5),static_cast<VertexCoord>(5),static_cast<VertexCoord>(0)};//static_cast<VertexCoord>(0),static_cast<VertexCoord>(0),static_cast<VertexCoord>(0)];

      mb.SetAttributeValuesForFace(position_att_id, draco::FaceIndex(tri_i), &temp,&temp1,&temp2);
                                    // &points[0], &points[1], &points[2]);
  }

  std::unique_ptr<draco::Mesh> draco_mesh = mb.Finalize();  
  draco::EncoderBuffer eb;
  draco::Encoder encoder;
  encoder.SetEncodingMethod(draco::MESH_EDGEBREAKER_ENCODING);
  encoder.SetAttributePredictionScheme(draco::GeometryAttribute::POSITION,
                                       draco::MESH_PREDICTION_PARALLELOGRAM);
  auto draco_status = encoder.EncodeMeshToBuffer(*draco_mesh, &eb);
  
  std::cout<<(draco_status.ok()==true)<<"\n";
  std::cout<<draco_status.error_msg_string()<<"\n";

  const std::string filename = "/groups/scicompsoft/home/ackermand/Programming/meshes/dracoC/temp.drc";

  std::cout<<eb.size() <<"\n";

  const void* cvp = eb.data(); 

  draco::FileWriterFactory::RegisterWriter(draco::StdioFileWriter::Open);
  draco::WriteBufferToFile(eb.data(), eb.size(), filename);

  return 0;
}*/

void ConvertToDracoMeshWithPreQuantization(std::string & directory, Vector3d & fragment_shape, int * fragment_position){
  Mesh mesh = ReadObjFile(directory,fragment_shape, fragment_position);
  int num_quantization_bits = 10;

  Quantizer quantizer(mesh.fragment_origin, mesh.fragment_shape,
                                     mesh.input_origin, num_quantization_bits);

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
    
    /*std::cout<<(draco_status.ok()==true)<<"\n";
    std::cout<<draco_status.error_msg_string()<<"\n";
    std::cout<<eb.size() <<"\n";

    const void* cvp = eb.data(); 
  */
    std::string output_filename=directory;
    for(int d=0; d<3; d++){
      output_filename+= std::to_string(fragment_position[d]);
    }
    output_filename+=".drc";
    draco::FileWriterFactory::RegisterWriter(draco::StdioFileWriter::Open);
    draco::WriteBufferToFile(eb.data(), eb.size(), output_filename);
  }

}

int main(int argc, char ** argv){

  std::string directory = argv[1];
  Vector3d fragment_shape = {std::strtof(argv[2],NULL),std::strtof(argv[3],NULL),std::strtof(argv[4],NULL)};
  int fragment_position[3]= {std::atoi(argv[5]),std::atoi(argv[6]),std::atoi(argv[7])};
  ConvertToDracoMeshWithPreQuantization(directory, fragment_shape, fragment_position);
    
  return 1;
}