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
ConfigureMobility(NodeContainer& ueNodes, const std::string& mobilityModel, double distance = 0.0)
{
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        double x = 0.0;
        if (mobilityModel == "Static" && distance > 0.0)
        {
            x = distance;
        }
        positionAlloc->Add(Vector(x, 0.0, 1.5));
    }

    mobility.SetPositionAllocator(positionAlloc);

    if (mobilityModel == "Static")
    {
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    }
    else if (mobilityModel == "Pedestrian")
    {
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
        mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                                  "Speed",
                                  StringValue("ns3::UniformRandomVariable[Min=15.0|Max=18.0]"),
                                  "Pause",
                                  StringValue("ns3::ConstantRandomVariable[Constant=0.1]"),
                                  "PositionAllocator",
                                  PointerValue(positionAlloc));
    }

    mobility.Install(ueNodes);
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
        lteHelper->SetSchedulerAttribute("CqiTimerThreshold", UintegerValue(1000));
        lteHelper->SetSchedulerAttribute("HarqEnabled", BooleanValue(true));
    }
    else if (schedulerType == "MT")
    {
        lteHelper->SetSchedulerType("ns3::FdMtFfMacScheduler");
        lteHelper->SetSchedulerAttribute("HarqEnabled", BooleanValue(true));
    }
    else if (schedulerType == "TB")
    {
        lteHelper->SetSchedulerType("ns3::FdTbfqFfMacScheduler");
    }
}

void
ConfigureStatsCollection(Ptr<LteHelper> lteHelper, const std::string& outputDir)
{
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

    Ptr<RadioBearerStatsCalculator> pdcpStats = lteHelper->GetPdcpStats();
    pdcpStats->SetAttribute("EpochDuration", TimeValue(Seconds(0.05)));
    pdcpStats->SetAttribute("DlPdcpOutputFilename", StringValue(outputDir + "/dl_pdcp_stats.csv"));
    pdcpStats->SetAttribute("UlPdcpOutputFilename", StringValue(outputDir + "/ul_pdcp_stats.csv"));

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
              std::string outputDir,
              double distance = 0.0)
{
    std::string folderName =
        outputDir + "/" + scenario + "_BW" + std::to_string(bandwidth) + "_UEs" +
        std::to_string(numUes) + "_TM" + std::to_string(transmissionMode) + "_CA" +
        (useCarrierAggregation ? "On" : "Off") + "_Sched" + schedulerType + "_Mob" + mobilityModel;

    if (mobilityModel == "Static" && distance > 0)
    {
        folderName += "_Dist" + std::to_string((int)distance);
    }

    CreateDirectory(folderName);

    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    ConfigureScheduler(lteHelper, schedulerType);

    lteHelper->SetAttribute("PathlossModel", StringValue("ns3::OkumuraHataPropagationLossModel"));

    lteHelper->SetPathlossModelAttribute(
        "Frequency",
        DoubleValue(800e6));
    if (scenario == "Urban")
    {
        lteHelper->SetPathlossModelAttribute("Environment",
                                             StringValue("Urban"));
        lteHelper->SetPathlossModelAttribute("CitySize",
                                             StringValue("Large"));
    }
    else
    {
        lteHelper->SetPathlossModelAttribute("Environment", StringValue("SubUrban"));
        lteHelper->SetPathlossModelAttribute("CitySize",
                                             StringValue("Small"));
    }
    ConfigureFadingModel(lteHelper, mobilityModel);

    Config::SetDefault("ns3::LteAmc::AmcModel", EnumValue(LteAmc::PiroEW2010));

    if (useCarrierAggregation)
    {
        Config::SetDefault("ns3::LteHelper::UseCa", BooleanValue(true));
        Config::SetDefault("ns3::LteHelper::NumberOfComponentCarriers", UintegerValue(2));
        Config::SetDefault("ns3::LteHelper::EnbComponentCarrierManager",
                           StringValue("ns3::RrComponentCarrierManager"));
    }

    NodeContainer enbNodes, ueNodes;
    enbNodes.Create(1);
    ueNodes.Create(numUes);

    MobilityHelper enbMobility;
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.Install(enbNodes);

    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
    enbPositionAlloc->Add(Vector(0.0, 0.0, 30.0));
    MobilityHelper enbMobilityHelper;
    enbMobilityHelper.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobilityHelper.SetPositionAllocator(enbPositionAlloc);
    enbMobilityHelper.Install(enbNodes);

    ConfigureMobility(ueNodes, mobilityModel, distance);
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(bandwidth));
    lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(bandwidth));

    for (uint16_t i = 0; i < ueLteDevs.GetN(); i++)
    {
        Ptr<LteUeNetDevice> ueDev = ueLteDevs.Get(i)->GetObject<LteUeNetDevice>();
        ueDev->GetPhy()->SetTransmissionMode(transmissionMode);
    }

    InternetStackHelper internet;
    internet.Install(ueNodes);

    Ipv4InterfaceContainer ueIpIfaces;
    for (uint16_t i = 0; i < numUes; i++)
    {
        lteHelper->Attach(ueLteDevs.Get(i), enbLteDevs.Get(0));
        ueIpIfaces.Add(epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs.Get(i))));
    }

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    ApplicationContainer serverApps, clientApps;
    uint16_t basePort = 50000;

    for (uint16_t i = 0; i < numUes; i++)
    {
        Ptr<NetDevice> ueDevice = ueLteDevs.Get(i);
        Ptr<EpcTft> tft = Create<EpcTft>();
        EpcTft::PacketFilter pf;
        uint16_t port = basePort + i;

        // Determine traffic type based on UE index modulo 4
        std::string trafficType;
        GbrQosInformation gbrQosInfo;
        EpsBearer::Qci qci;

        switch (i % 4)
        {
        case 0: // Voice (25% of UEs)
            trafficType = "voice";
            gbrQosInfo.gbrDl =
                300e3; // 300 Kbps
                       // https://www.researchgate.net/publication/290773895_QoS-aware_resource_management_for_LTE-Advanced_relay-enhanced_network
            gbrQosInfo.mbrDl = 500e3;
            qci = EpsBearer::GBR_CONV_VOICE;
            break;
        case 1: // Video (25% of UEs)
            trafficType = "video";
            gbrQosInfo.gbrDl =
                75e5; // 7.5 Mbps
                      // https://www.researchgate.net/publication/290773895_QoS-aware_resource_management_for_LTE-Advanced_relay-enhanced_network
            gbrQosInfo.mbrDl = 75e5;
            qci = EpsBearer::GBR_CONV_VIDEO;
            break;
        case 2: // Gaming (25% of UEs)
            trafficType = "gaming";
            gbrQosInfo.gbrDl =
                25e5; // 2.5 Mbps
                      // https://www.researchgate.net/publication/290773895_QoS-aware_resource_management_for_LTE-Advanced_relay-enhanced_network
            gbrQosInfo.mbrDl = 25e5;
            qci = EpsBearer::GBR_GAMING;
            break;
        case 3: // Best-effort NGBR traffic (25% of UEs)
            trafficType = "best-effort";
            gbrQosInfo.gbrDl = 0;
            gbrQosInfo.mbrDl = 0;
            qci = EpsBearer::NGBR_VIDEO_TCP_DEFAULT;
            break;
        }

        pf.direction = EpcTft::DOWNLINK;
        pf.remotePortStart = port;
        pf.remotePortEnd = port;
        tft->Add(pf);
        lteHelper->ActivateDedicatedEpsBearer(ueDevice, EpsBearer(qci, gbrQosInfo), tft);

        if (qci != EpsBearer::NGBR_VIDEO_TCP_DEFAULT)
        {
            PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                        InetSocketAddress(Ipv4Address::GetAny(), port));
            OnOffHelper onOffHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(ueIpIfaces.GetAddress(i), port));

            onOffHelper.SetAttribute("DataRate", DataRateValue(DataRate(gbrQosInfo.gbrDl)));
            onOffHelper.SetAttribute("PacketSize", UintegerValue(1400));
            onOffHelper.SetAttribute("OnTime",
                                     StringValue("ns3::ConstantRandomVariable[Constant=1]"));
            onOffHelper.SetAttribute("OffTime",
                                     StringValue("ns3::ConstantRandomVariable[Constant=0]"));

            serverApps.Add(sinkHelper.Install(ueNodes.Get(i)));
            clientApps.Add(onOffHelper.Install(pgw));
        }
        else
        {
            PacketSinkHelper sinkHelper("ns3::TcpSocketFactory",
                                        InetSocketAddress(Ipv4Address::GetAny(), port));
            BulkSendHelper onOffHelper("ns3::TcpSocketFactory",
                                       InetSocketAddress(ueIpIfaces.GetAddress(i), port));

            onOffHelper.SetAttribute("MaxBytes", UintegerValue(0));
            onOffHelper.SetAttribute("SendSize", UintegerValue(1400));

            serverApps.Add(sinkHelper.Install(ueNodes.Get(i)));
            clientApps.Add(onOffHelper.Install(pgw));
        }
    }

    serverApps.Start(Seconds(0.1));
    serverApps.Stop(Seconds(10.0));
    clientApps.Start(Seconds(0.1));
    clientApps.Stop(Seconds(10.0));

    ConfigureStatsCollection(lteHelper, folderName);

    Simulator::Stop(Seconds(10.0));
    Simulator::Run();

    Simulator::Destroy();
}

int
main(int argc, char* argv[])
{
    std::vector<uint16_t> bandwidths = {25, 50, 75, 100};
    std::vector<uint16_t> ueNumbers = {1, 4, 8, 10};
    std::vector<uint8_t> transmissionModes = {0, 3};
    std::vector<bool> carrierAggregationOptions = {false, true};
    std::vector<std::string> schedulerTypes = {"RR", "PF", "CQA", "MT", "TB"};
    std::vector<std::string> mobilityModels = {"Static", "Pedestrian", "Vehicle"};
    std::vector<double> staticDistances = {10, 50, 100};
    std::vector<std::string> scenarios = {"Urban", "Suburban"};

    std::string outputDir = "SimulationResults";
    CreateDirectory(outputDir);

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
                                if (mobilityModel == "Static")
                                {
                                    for (auto distance : staticDistances)
                                    {
                                        NS_LOG_UNCOND("Running static simulation at distance "
                                                      << distance << "m: "
                                                      << "BW=" << bw << "MHz, "
                                                      << "UEs=" << ueNum << ", "
                                                      << "TM=" << (int)tm << ", "
                                                      << "CA=" << (ca ? "On" : "Off") << ", "
                                                      << "Scheduler=" << schedulerType);

                                        RunSimulation(bw,
                                                      ueNum,
                                                      tm,
                                                      ca,
                                                      schedulerType,
                                                      mobilityModel,
                                                      scenario,
                                                      outputDir,
                                                      distance);
                                    }
                                }
                                else
                                {
                                    NS_LOG_UNCOND("Running mobile simulation: "
                                                  << "BW=" << bw << "MHz, "
                                                  << "UEs=" << ueNum << ", "
                                                  << "TM=" << (int)tm << ", "
                                                  << "CA=" << (ca ? "On" : "Off") << ", "
                                                  << "Scheduler=" << schedulerType << ", "
                                                  << "Mobility=" << mobilityModel);

                                    RunSimulation(bw,
                                                  ueNum,
                                                  tm,
                                                  ca,
                                                  schedulerType,
                                                  mobilityModel,
                                                  scenario,
                                                  outputDir);
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