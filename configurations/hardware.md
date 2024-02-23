# QCFE Hardware Configuration

```
{
  "DeviceList": [
    {
      "DeviceName": "Device0",
      "Manufacturer": "Lenovo",
      "HardwareSpecifications": {
        "Processor": "AMD Ryzen™ 7 4800U",
        "Memory": "8GB 3200MHz (Micron) + 8GB 3200MHz (Micron)",
        "Storage": {
          "Type": "SSD",
          "Capacity": "512 GB",
          "Manufacturer": "SAMSUNG",
          "Model": "MZVLB512HBJQ-000L2",
          "Interface Protocol": "PCI-E 3.0x4 NVMe"
        },
        "GraphicsProcessor": "AMD Radeon™ 512MB Graphics"
      },
      "knob_configuration":[1.conf-20.conf] # all in knob_configuration dir
    },
    {
      "DeviceName": "Device1",
      "Manufacturer": "Lenovo",
      "Model": "Legion Y9000X IAH7",
      "HardwareSpecifications": {
        "Processor": "Intel Core™ i7 12700H",
        "Memory": "32 GB 4800MHz (Crucial) + 8GB 4800MHz (Samsung)",
        "StorageList": [
        {
          "Type": "SSD",
          "Capacity": "512 GB",
          "Manufacturer": "Micron",
          "Model": "MTFDKBA512TFH",
          "Interface Protocol": "PCI-E 3.0x4 NVMe"
        },
        {
          "Type": "SSD",
          "Capacity": "2 TB",
          "Manufacturer": "Crucial",
          "Model": "CT2000P2SSD8",
          "Interface Protocol": "PCI-E 3.0x4 NVMe"
        },
        ]
        "GraphicsProcessor": "NVIDIA GeForce RTX 3060 Laptop 6GB"
      },
      "knob_configuration":[17.conf, 18.conf, 19.conf]
    },
    {
      "DeviceName": "Device2",
      "HardwareSpecifications": {
        "Processor": "OctalCore Intel Core™ i7-10700, 4200 MHz (42 x 100)",
        "Motherboard": "Asus Prime Z490-P  (4 PCI-E x1, 2 PCI-E x16, 3 M.2, 4 DDR4 DIMM, Audio, Video, Gigabit LAN, Thunderbolt 3)",
        "Chipset": "Intel Comet Point-H Z490, Intel Comet Lake-S",
        "Memory": {
          "Model": "16GBx2 Corsair CM4X16GC3200C16K2E",
          "Memory Timings": "C16",
          "Frequency": "3200MHz",
          "DRAM Manufacturer": "Micron"
        },
        "StorageList": [
            {
              "Type": "HDD",
              "Capacity": "2 TB",
              "Manufacturer": "Seagate",
              "Model": "ST2000DM005-2CW102",
              "Rotation Speed": "5000RPM",
              "Interface Protocol": "SATA-III"
            },
            {
              "Type": "SSD",
              "Capacity": "500 GB",
              "Manufacturer": "Western Digital",
              "Model": "WDS500G3X0C-00SJG0",
              "Series": "SN750",
              "Interface Protocol": "PCI-E 3.0x4 NVMe"
            }
        ]
        "GraphicsProcessor": {
        	"Display Adapter": "nVIDIA GeForce RTX 3060 LHR",
    		"GPU Code Name": "GA106-302 [LHR]",
    		"PCI Device": "10DE-2504 / 7377-2000 (Rev A1)",
    		"Process Technology": "8 nm",
    		"Bus Type": "PCI Express 3.0 x16 @ 1.1 x16",
    		"Memory Size": "12 GB",
    		"Bus Type": "GDDR6 (Samsung)",
    		"Bus Width": 192,
    		"Bandwidth": "19.0 GB/s"
        },
      "knob_configuration":[17.conf, 18.conf, 19.conf]
    },
  ]
}
```

