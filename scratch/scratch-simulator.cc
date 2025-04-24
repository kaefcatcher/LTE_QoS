#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/propagation-module.h"

#include <fstream>
#include <iomanip>
#include <sys/stat.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LteQosSimulation");

void
CreateDirectory(const std::string& path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        if (mkdir(path.c_str(), 0777) != 0)
        {
            NS_FATAL_ERROR("Failed to create directory: " << path);
        }
    }
}

void
ConfigureFadingModel(Ptr<LteHelper> lteHelper, const std::string& mobilityModel)
{
    if (mobilityModel == "Static")
    {
        // No fading for static scenario
        return;
    }
    else if (mobilityModel == "Pedestrian")
    {
        lteHelper->SetFadingModel("ns3::TraceFadingLossModel");
        lteHelper->SetFadingModelAttribute(
            "TraceFilename",
            StringValue("src/lte/model/fading-traces/fading_trace_EPA_3kmph.fad"));
        lteHelper->SetFadingModelAttribute("TraceLength", TimeValue(Seconds(10.0)));
        lteHelper->SetFadingModelAttribute("SamplesNum", UintegerValue(10000));
        lteHelper->SetFadingModelAttribute("WindowSize", TimeValue(Seconds(0.5)));
        lteHelper->SetFadingModelAttribute("RbNum", UintegerValue(100));
    }
    else if (mobilityModel == "Vehicle")
    {
        lteHelper->SetFadingModel("ns3::TraceFadingLossModel");
        lteHelper->SetFadingModelAttribute(
            "TraceFilename",
            StringValue("src/lte/model/fading-traces/fading_trace_EVA_60kmph.fad"));
        lteHelper->SetFadingModelAttribute("TraceLength", TimeValue(Seconds(10.0)));
        lteHelper->SetFadingModelAttribute("SamplesNum", UintegerValue(10000));
        lteHelper->SetFadingModelAttribute("WindowSize", TimeValue(Seconds(0.5)));
        lteHelper->SetFadingModelAttribute("RbNum", UintegerValue(100));
    }
}

void
ConfigureUrbanPropagationModel(Ptr<LteHelper> lteHelper)
{
    // Urban propagation model - COST231 with configurable parameters
    lteHelper->SetAttribute("PathlossModel", StringValue("ns3::Cost231PropagationLossModel"));

    // Set COST231 parameters with realistic defaults for urban LTE
    lteHelper->SetPathlossModelAttribute("Frequency", DoubleValue(2000e6)); // 2GHz
    lteHelper->SetPathlossModelAttribute("Lambda", DoubleValue(300000000.0 / 2000e6));
    lteHelper->SetPathlossModelAttribute("BSAntennaHeight", DoubleValue(30.0)); // 30m eNB height
    lteHelper->SetPathlossModelAttribute("SSAntennaHeight", DoubleValue(1.5));  // 1.5m UE height
    lteHelper->SetPathlossModelAttribute("MinDistance",
                                         DoubleValue(10.0)); // Minimum valid distance

    // Add log-distance shadowing for urban environment
    // lteHelper->SetPathlossModelAttribute("ShadowSigmaOutdoor", DoubleValue(8.0)); // 8dB stddev
    // lteHelper->SetPathlossModelAttribute("ShadowSigmaIndoor", DoubleValue(10.0)); // 10dB stddev
}

void
ConfigureSuburbanPropagationModel(Ptr<LteHelper> lteHelper)
{
    // Suburban propagation model - OkumuraHata with configurable parameters
    lteHelper->SetAttribute("PathlossModel", StringValue("ns3::FriisSpectrumPropagationLossModel"));

    // Set Okumura-Hata parameters with realistic defaults for suburban LTE
    lteHelper->SetPathlossModelAttribute("Frequency", DoubleValue(2000e6)); // 2GHz
    // lteHelper->SetPathlossModelAttribute("Environment", EnumValue(SubUrbanEnvironment));
    // lteHelper->SetPathlossModelAttribute("CitySize", EnumValue(LargeCity));

    // Antenna heights
    lteHelper->SetPathlossModelAttribute("BSAntennaHeight", DoubleValue(30.0)); // 30m eNB height
    lteHelper->SetPathlossModelAttribute("SSAntennaHeight", DoubleValue(1.5));  // 1.5m UE height

    lteHelper->SetPathlossModelAttribute("MinDistance",
                                         DoubleValue(50.0)); // Minimum valid distance

    // Add log-distance shadowing for suburban environment
    // lteHelper->SetPathlossModelAttribute("ShadowSigmaOutdoor", DoubleValue(6.0)); // 6dB stddev
    // lteHelper->SetPathlossModelAttribute("ShadowSigmaIndoor", DoubleValue(8.0));  // 8dB stddev
}

void
ConfigurePropagationModel(Ptr<LteHelper> lteHelper, const std::string& scenario)
{
    if (scenario == "Urban")
    {
        ConfigureUrbanPropagationModel(lteHelper);
    }
    else if (scenario == "Suburban")
    {
        ConfigureSuburbanPropagationModel(lteHelper);
    }

    // Common configuration for both models
    lteHelper->SetSpectrumChannelType("ns3::MultiModelSpectrumChannel");

    // Enable realistic fast fading
    lteHelper->SetFadingModel("ns3::TraceFadingLossModel");
    if (scenario == "Urban")
    {
        lteHelper->SetFadingModelAttribute(
            "TraceFilename",
            StringValue("src/lte/model/fading-traces/fading_trace_ETU_3kmph.fad"));
    }
    else
    {
        lteHelper->SetFadingModelAttribute(
            "TraceFilename",
            StringValue("src/lte/model/fading-traces/fading_trace_EPA_3kmph.fad"));
    }
    lteHelper->SetFadingModelAttribute("TraceLength", TimeValue(Seconds(10.0)));
    lteHelper->SetFadingModelAttribute("SamplesNum", UintegerValue(10000));
    lteHelper->SetFadingModelAttribute("WindowSize", TimeValue(Seconds(0.5)));
    lteHelper->SetFadingModelAttribute("RbNum", UintegerValue(100));
}

void
ConfigureMobility(NodeContainer& ueNodes, const std::string& mobilityModel, double distance = 0.0)
{
    MobilityHelper mobility;

    if (mobilityModel == "Static")
    {
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

        // Position UEs at specified distance from eNB if distance > 0
        if (distance > 0.0)
        {
            Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

            // Place eNB at (0,0,30)
            // Place UEs along x-axis at specified distance, height=1.5m (typical UE height)
            for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
            {
                positionAlloc->Add(Vector(distance, 0.0, 1.5));
            }
            mobility.SetPositionAllocator(positionAlloc);
        }
    }
    else if (mobilityModel == "Pedestrian")
    {
        // 3 km/h = 0.833 m/s
        mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                  "Bounds",
                                  RectangleValue(Rectangle(-500, 500, -500, 500)),
                                  "Speed",
                                  StringValue("ns3::ConstantRandomVariable[Constant=0.833]"),
                                  "Distance",
                                  DoubleValue(100));
    }
    else if (mobilityModel == "Vehicle")
    {
        // 60 km/h = 16.667 m/s
        mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                                  "Speed",
                                  StringValue("ns3::UniformRandomVariable[Min=15.0|Max=18.0]"),
                                  "Pause",
                                  StringValue("ns3::ConstantRandomVariable[Constant=0.1]"),
                                  "PositionAllocator",
                                  PointerValue(CreateObject<RandomRectanglePositionAllocator>()));
    }

    mobility.Install(ueNodes);
}

EpsBearer::Qci
GetQciForTrafficType(const std::string& trafficType, bool isGbr)
{
    if (isGbr)
    {
        if (trafficType == "voice")
            return EpsBearer::GBR_CONV_VOICE;
        if (trafficType == "video")
            return EpsBearer::GBR_CONV_VIDEO;
        if (trafficType == "gaming")
            return EpsBearer::GBR_GAMING;
    }
    else
    {
        if (trafficType == "voice")
            return EpsBearer::NGBR_VOICE_VIDEO_GAMING;
        if (trafficType == "video")
            return EpsBearer::NGBR_VIDEO_TCP_DEFAULT;
        if (trafficType == "gaming")
            return EpsBearer::NGBR_VOICE_VIDEO_GAMING;
    }
    return EpsBearer::NGBR_VIDEO_TCP_DEFAULT;
}

void ConfigureTrafficApplications(NodeContainer& ueNodes,
                                 Ptr<Node>& pgw,
                                 Ipv4InterfaceContainer& ueIpIfaces,
                                 const std::string& gbrTrafficType,
                                 const std::string& nonGbrTrafficType)
{
    // GBR traffic configuration (UDP)
    uint16_t gbrPort = 50000;
    ApplicationContainer gbrServerApps, gbrClientApps;

    // Non-GBR traffic configuration (TCP with proper config)
    uint16_t nonGbrPort = 50001;
    ApplicationContainer nonGbrServerApps, nonGbrClientApps;

    // Configure TCP parameters
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(1448));
    Config::SetDefault("ns3::TcpSocket::SndBufSize", UintegerValue(131072));
    Config::SetDefault("ns3::TcpSocket::RcvBufSize", UintegerValue(131072));

    for (uint32_t i = 0; i < ueNodes.GetN(); i++)
    {
        // GBR traffic (UDP)
        PacketSinkHelper gbrPacketSinkHelper("ns3::UdpSocketFactory",
                                         InetSocketAddress(Ipv4Address::GetAny(), gbrPort));
        gbrServerApps.Add(gbrPacketSinkHelper.Install(ueNodes.Get(i)));

        OnOffHelper gbrOnOffHelper("ns3::UdpSocketFactory",
                                 InetSocketAddress(ueIpIfaces.GetAddress(i), gbrPort));
        gbrOnOffHelper.SetAttribute("PacketSize", UintegerValue(1400));
        
        // Higher data rates
        if (gbrTrafficType == "voice") {
            gbrOnOffHelper.SetAttribute("DataRate", DataRateValue(DataRate("1Mbps")));
        } else if (gbrTrafficType == "video") {
            gbrOnOffHelper.SetAttribute("DataRate", DataRateValue(DataRate("10Mbps")));
        } else { // gaming
            gbrOnOffHelper.SetAttribute("DataRate", DataRateValue(DataRate("5Mbps")));
        }
        
        gbrOnOffHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        gbrOnOffHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        gbrClientApps.Add(gbrOnOffHelper.Install(pgw));

        // Non-GBR traffic (TCP)
        PacketSinkHelper nonGbrPacketSinkHelper("ns3::TcpSocketFactory",
                                             InetSocketAddress(Ipv4Address::GetAny(), nonGbrPort));
        nonGbrServerApps.Add(nonGbrPacketSinkHelper.Install(ueNodes.Get(i)));

        BulkSendHelper nonGbrBulkHelper("ns3::TcpSocketFactory",
                                     InetSocketAddress(ueIpIfaces.GetAddress(i), nonGbrPort));
        nonGbrBulkHelper.SetAttribute("SendSize", UintegerValue(1448));
        nonGbrBulkHelper.SetAttribute("MaxBytes", UintegerValue(0));
        nonGbrClientApps.Add(nonGbrBulkHelper.Install(pgw));

        gbrPort++;
        nonGbrPort++;
    }

    // Longer simulation time
    gbrServerApps.Start(Seconds(0.1));
    gbrServerApps.Stop(Seconds(10.0));
    gbrClientApps.Start(Seconds(0.1));
    gbrClientApps.Stop(Seconds(10.0));

    nonGbrServerApps.Start(Seconds(0.1));
    nonGbrServerApps.Stop(Seconds(10.0));
    nonGbrClientApps.Start(Seconds(0.1));
    nonGbrClientApps.Stop(Seconds(10.0));
}

void
ConfigureScheduler(Ptr<LteHelper> lteHelper, const std::string& schedulerType)
{
    if (schedulerType == "RR")
    {
        lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    }
    else if (schedulerType == "PF")
    {
        lteHelper->SetSchedulerType("ns3::PfFfMacScheduler");
    }
    else if (schedulerType == "CQA")
    {
        lteHelper->SetSchedulerType("ns3::CqaFfMacScheduler");
        // Configure QoS-aware parameters
        lteHelper->SetSchedulerAttribute("CqiTimerThreshold", UintegerValue(1000));
        lteHelper->SetSchedulerAttribute("HarqEnabled", BooleanValue(true));
    }
    else if (schedulerType == "MT")
    {
        lteHelper->SetSchedulerType("ns3::FdMtFfMacScheduler");
        // Configure max throughput parameters
        lteHelper->SetSchedulerAttribute("HarqEnabled", BooleanValue(true));
    }
}

void
ConfigureStatsCollection(Ptr<LteHelper> lteHelper, const std::string& outputDir)
{
    // Enable all trace sources
    lteHelper->EnableTraces();

    Ptr<PhyTxStatsCalculator> phyTxStats = lteHelper->GetPhyTxStats();
    phyTxStats->SetAttribute("DlTxOutputFilename", StringValue(outputDir + "/dl_tx_phy_stats.csv"));
    phyTxStats->SetAttribute("UlTxOutputFilename", StringValue(outputDir + "/ul_tx_phy_stats.csv"));

    Ptr<PhyRxStatsCalculator> phyRxStats = lteHelper->GetPhyRxStats();
    phyRxStats->SetAttribute("DlRxOutputFilename", StringValue(outputDir + "/dl_rx_phy_stats.csv"));
    phyRxStats->SetAttribute("UlRxOutputFilename", StringValue(outputDir + "/ul_rx_phy_stats.csv"));

    Ptr<MacStatsCalculator> macStats = lteHelper->GetMacStats();
    macStats->SetAttribute("DlOutputFilename", StringValue(outputDir + "/dl_mac_stats.csv"));
    macStats->SetAttribute("UlOutputFilename", StringValue(outputDir + "/ul_mac_stats.csv"));

    Ptr<RadioBearerStatsCalculator> rlcStats = lteHelper->GetRlcStats();
    rlcStats->SetAttribute("EpochDuration", TimeValue(Seconds(0.05)));
    rlcStats->SetAttribute("DlRlcOutputFilename", StringValue(outputDir + "/dl_rlc_stats.csv"));
    rlcStats->SetAttribute("UlRlcOutputFilename", StringValue(outputDir + "/ul_rlc_stats.csv"));

    Ptr<RadioBearerStatsCalculator> pdcpStats = lteHelper->GetPdcpStats();
    pdcpStats->SetAttribute("EpochDuration", TimeValue(Seconds(0.05)));
    pdcpStats->SetAttribute("DlPdcpOutputFilename", StringValue(outputDir + "/dl_pdcp_stats.csv"));
    pdcpStats->SetAttribute("UlPdcpOutputFilename", StringValue(outputDir + "/ul_pdcp_stats.csv"));

    // Radio Environment Map (for SINR visualization)
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));
    lteHelper->SetAttribute("UseIdealRrc", BooleanValue(true));
}

void
RunSimulation(uint16_t bandwidth,
              uint16_t numUes,
              uint8_t transmissionMode,
              bool useCarrierAggregation,
              std::string schedulerType,
              std::string mobilityModel,
              std::string scenario,
              std::string gbrTrafficType,
              std::string nonGbrTrafficType,
              std::string outputDir,
              double distance = 0.0)
{
    // Update folder name to include scenario
    std::string folderName =
        outputDir + "/" + scenario + "_BW" + std::to_string(bandwidth) + "_UEs" +
        std::to_string(numUes) + "_TM" + std::to_string(transmissionMode) + "_CA" +
        (useCarrierAggregation ? "On" : "Off") + "_Sched" + schedulerType + "_Mob" + mobilityModel;

    if (mobilityModel == "Static" && distance > 0)
    {
        folderName += "_Dist" + std::to_string((int)distance);
    }

    folderName += "_GBR" + gbrTrafficType + "_NonGBR" + nonGbrTrafficType;

    CreateDirectory(folderName);

    // Create LTE helper and configure propagation model
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    ConfigureScheduler(lteHelper, schedulerType);

    // Configure propagation and fading models
    ConfigurePropagationModel(lteHelper, scenario);
    ConfigureFadingModel(lteHelper, mobilityModel);

    // Configure realistic MCS (not always max)
    Config::SetDefault("ns3::LteAmc::AmcModel", EnumValue(LteAmc::PiroEW2010));

    // Configure carrier aggregation if enabled
    if (useCarrierAggregation)
    {
        Config::SetDefault("ns3::LteHelper::UseCa", BooleanValue(true));
        Config::SetDefault("ns3::LteHelper::NumberOfComponentCarriers", UintegerValue(2));
        Config::SetDefault("ns3::LteHelper::EnbComponentCarrierManager",
                           StringValue("ns3::RrComponentCarrierManager"));
    }

    // Create nodes
    NodeContainer enbNodes, ueNodes;
    enbNodes.Create(1);
    ueNodes.Create(numUes);

    // Configure mobility - now passing distance parameter
    MobilityHelper enbMobility;
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.Install(enbNodes);

    // Position eNB at origin
    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
    enbPositionAlloc->Add(Vector(0.0, 0.0, 30.0)); // eNB at (0,0,30)
    MobilityHelper enbMobilityHelper;
    enbMobilityHelper.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobilityHelper.SetPositionAllocator(enbPositionAlloc);
    enbMobilityHelper.Install(enbNodes);

    ConfigureMobility(ueNodes, mobilityModel, distance);
    // Install LTE devices
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    // Set bandwidth and transmission mode
    lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(bandwidth));
    lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(bandwidth));

    for (uint16_t i = 0; i < ueLteDevs.GetN(); i++)
    {
        Ptr<LteUeNetDevice> ueDev = ueLteDevs.Get(i)->GetObject<LteUeNetDevice>();
        ueDev->GetPhy()->SetTransmissionMode(transmissionMode);
    }

    // Install IP stack
    InternetStackHelper internet;
    internet.Install(ueNodes);

    // Assign IP addresses to UEs
    Ipv4InterfaceContainer ueIpIfaces;
    for (uint16_t i = 0; i < numUes; i++)
    {
        lteHelper->Attach(ueLteDevs.Get(i), enbLteDevs.Get(0));
        ueIpIfaces.Add(epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs.Get(i))));
    }

    // Setup bearers
    EpsBearer::Qci gbrQci = GetQciForTrafficType(gbrTrafficType, true);
    EpsBearer::Qci nonGbrQci = GetQciForTrafficType(nonGbrTrafficType, false);

    for (uint16_t i = 0; i < numUes; i++)
    {
        Ptr<LteUeNetDevice> ueLteDev = ueLteDevs.Get(i)->GetObject<LteUeNetDevice>();
        if (i%2 == 0)
        lteHelper->ActivateDedicatedEpsBearer(ueLteDev, EpsBearer(gbrQci), EpcTft::Default());
        else
        lteHelper->ActivateDedicatedEpsBearer(ueLteDev, EpsBearer(nonGbrQci), EpcTft::Default());
    }

    // Configure traffic applications
    Ptr<Node> pgw = epcHelper->GetPgwNode();
    ConfigureTrafficApplications(ueNodes, pgw, ueIpIfaces, gbrTrafficType, nonGbrTrafficType);

    // Configure statistics collection
    ConfigureStatsCollection(lteHelper, folderName);

    // Run simulation
    Simulator::Stop(Seconds(10.0));
    Simulator::Run();

    Simulator::Destroy();
}

int
main(int argc, char* argv[])
{
    // Simulation parameters
    std::vector<uint16_t> bandwidths = {6, 15, 25, 50, 100};
    std::vector<uint16_t> ueNumbers = {1, 2, 5, 10};
    std::vector<uint8_t> transmissionModes = {0, 3};
    std::vector<bool> carrierAggregationOptions = {false, true};
    std::vector<std::string> schedulerTypes = {"RR", "PF", "CQA", "MT"};
    std::vector<std::string> mobilityModels = {"Static", "Pedestrian", "Vehicle"};
    std::vector<std::string> gbrTrafficTypes = {"video"};
    std::vector<std::string> nonGbrTrafficTypes = {"voice"};
    std::vector<double> staticDistances = {50, 100, 200};
    std::vector<std::string> scenarios = {"Urban", "Suburban"};

    // Create output directory
    std::string outputDir = "SimulationResults";
    CreateDirectory(outputDir);

    // Run simulations for all parameter combinations
    for (auto scenario : scenarios)
    {
        for (auto bw : bandwidths)
        {
            for (auto ueNum : ueNumbers)
            {
                for (auto tm : transmissionModes)
                {
                    for (auto ca : carrierAggregationOptions)
                    {
                        for (auto schedulerType : schedulerTypes)
                        {
                            for (auto mobilityModel : mobilityModels)
                            {
                                // Handle static scenarios with different distances
                                if (mobilityModel == "Static")
                                {
                                    for (auto distance : staticDistances)
                                    {
                                        for (auto gbrTrafficType : gbrTrafficTypes)
                                        {
                                            for (auto nonGbrTrafficType : nonGbrTrafficTypes)
                                            {
                                                NS_LOG_UNCOND(
                                                    "Running static simulation at distance "
                                                    << distance << "m: "
                                                    << "BW=" << bw << "MHz, "
                                                    << "UEs=" << ueNum << ", "
                                                    << "TM=" << (int)tm << ", "
                                                    << "CA=" << (ca ? "On" : "Off") << ", "
                                                    << "Scheduler=" << schedulerType << ", "
                                                    << "GBR=" << gbrTrafficType << ", "
                                                    << "NonGBR=" << nonGbrTrafficType);

                                                RunSimulation(bw,
                                                              ueNum,
                                                              tm,
                                                              ca,
                                                              schedulerType,
                                                              mobilityModel,
                                                              scenario,
                                                              gbrTrafficType,
                                                              nonGbrTrafficType,
                                                              outputDir,
                                                              distance);
                                            }
                                        }
                                    }
                                }
                                else
                                { // Non-static scenarios
                                    for (auto gbrTrafficType : gbrTrafficTypes)
                                    {
                                        for (auto nonGbrTrafficType : nonGbrTrafficTypes)
                                        {
                                            NS_LOG_UNCOND("Running mobile simulation: "
                                                          << "BW=" << bw << "MHz, "
                                                          << "UEs=" << ueNum << ", "
                                                          << "TM=" << (int)tm << ", "
                                                          << "CA=" << (ca ? "On" : "Off") << ", "
                                                          << "Scheduler=" << schedulerType << ", "
                                                          << "Mobility=" << mobilityModel << ", "
                                                          << "GBR=" << gbrTrafficType << ", "
                                                          << "NonGBR=" << nonGbrTrafficType);

                                            RunSimulation(bw,
                                                          ueNum,
                                                          tm,
                                                          ca,
                                                          schedulerType,
                                                          mobilityModel,
                                                          scenario,
                                                          gbrTrafficType,
                                                          nonGbrTrafficType,
                                                          outputDir);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}