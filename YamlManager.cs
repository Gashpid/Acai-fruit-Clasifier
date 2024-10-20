using System;
using System.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using YamlDotNet.RepresentationModel;
using System.Collections.Generic;

namespace AcaiFruitClasifier
{
    public class YamlManager
    {
        public static void WriteYaml(string filePath, string tag, string value)
        {
            Dictionary<string, object> yamlData;

            if (File.Exists(filePath))
            {
                var yamlText = File.ReadAllText(filePath);

                var deserializer = new DeserializerBuilder()
                    .WithNamingConvention(CamelCaseNamingConvention.Instance)
                    .Build();

                try
                {
                    yamlData = deserializer.Deserialize<Dictionary<string, object>>(yamlText) ?? new Dictionary<string, object>();
                }
                catch (Exception)
                {
                    yamlData = new Dictionary<string, object>();
                }
            }
            else
            {
                yamlData = new Dictionary<string, object>();
            }

            if (!yamlData.ContainsKey(tag))
            {
                yamlData.Add(tag, value);
            }
            else
            {
                yamlData[tag] = value;
            }

            var serializer = new SerializerBuilder()
                .WithNamingConvention(CamelCaseNamingConvention.Instance)
                .Build();

            var yamlOutput = serializer.Serialize(yamlData);

            File.WriteAllText(filePath, yamlOutput);
        }

        public static string ReadYaml(string filePath, string tag)
        {
            if (File.Exists(filePath))
            {
                var yamlText = File.ReadAllText(filePath);

                var deserializer = new DeserializerBuilder()
                    .WithNamingConvention(CamelCaseNamingConvention.Instance)
                    .Build();

                try
                {
                    var yamlData = deserializer.Deserialize<Dictionary<string, object>>(yamlText);
                    if (yamlData != null && yamlData.TryGetValue(tag, out var value))
                    {
                        return value as string;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error al leer el archivo YAML: {ex.Message}");
                }
            }
            else
            {
                Console.WriteLine($"ERROR: El archivo {filePath} no existe.");
            }

            return null;
        }
    }
}
