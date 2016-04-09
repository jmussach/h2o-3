package water.parser;

import org.apache.avro.Schema;

import java.util.List;

import water.fvec.Vec;

/**
 * Utilities to work with Avro schema.
 */
public final class AvroUtil {

  /** Return true if the given schema can be transformed
   * into h2o type.
   *
   * @param s  avro field schema
   * @return  true if the schema can be transformed into H2O type
   */
  public static boolean isSupportedSchema(Schema s) {
    Schema.Type typ =  s.getType();
    switch (typ) {
      case BOOLEAN:
      case INT:
      case LONG:
      case FLOAT:
      case DOUBLE:
      case ENUM:
      case STRING:
      case NULL:
        return true;
      case UNION: // Flattenize the union
        List<Schema> unionSchemas = s.getTypes();
        if (unionSchemas.size() == 1) {
          return isSupportedSchema(unionSchemas.get(0));
        } else if (unionSchemas.size() == 2) {
          Schema s1 = unionSchemas.get(0);
          Schema s2 = unionSchemas.get(1);
          return s1.getType().equals(Schema.Type.NULL) && isSupportedSchema(s2)
                 || s2.getType().equals(Schema.Type.NULL) && isSupportedSchema(s1);
        }
      default:
        return false;
    }
  }

  /**
   * Transform Avro schema into H2O type.
   *
   * @param s  avro schema
   * @return  a byte representing H2O column type
   * @throws IllegalArgumentException  if schema is not supported
   */
  public static byte schemaToColumnType(Schema s) {
    Schema.Type typ =  s.getType();
    switch (typ) {
      case BOOLEAN:
      case INT:
      case LONG:
      case FLOAT:
      case DOUBLE:
        return Vec.T_NUM;
      case ENUM:
        return Vec.T_CAT;
      case STRING:
        return Vec.T_STR;
      case NULL:
        return Vec.T_BAD;
      case UNION: // Flattenize the union
        List<Schema> unionSchemas = s.getTypes();
        if (unionSchemas.size() == 1) {
          return schemaToColumnType(unionSchemas.get(0));
        } else if (unionSchemas.size() == 2) {
          Schema s1 = unionSchemas.get(0);
          Schema s2 = unionSchemas.get(1);
          if (s1.getType().equals(Schema.Type.NULL)) return schemaToColumnType(s2);
          else if (s2.getType().equals(Schema.Type.NULL)) return schemaToColumnType(s1);
        }
      default:
        throw new IllegalArgumentException("Unsupported Avro schema type: " + s);
    }
  }

  /**
   * Transform Avro schema into its primitive representation.
   *
   * @param s  avro schema
   * @return  primitive type as a result of transformation
   * @throws IllegalArgumentException if the schema has no primitive transformation
   */
  public static Schema.Type toPrimitiveType(Schema s) {
    Schema.Type typ = s.getType();
    switch(typ) {
      case BOOLEAN:
      case INT:
      case LONG:
      case FLOAT:
      case DOUBLE:
      case ENUM:
      case STRING:
      case NULL:
        return typ;
      case UNION:
        List<Schema> unionSchemas = s.getTypes();
        if (unionSchemas.size() == 1) {
          return toPrimitiveType(unionSchemas.get(0));
        } else if (unionSchemas.size() == 2) {
          Schema s1 = unionSchemas.get(0);
          Schema s2 = unionSchemas.get(1);
          if (s1.getType().equals(Schema.Type.NULL)) return toPrimitiveType(s2);
          else if (s2.getType().equals(Schema.Type.NULL)) return toPrimitiveType(s1);
        }
      default:
        throw new IllegalArgumentException("Unsupported Avro schema type: " + s);
    }
  }
}