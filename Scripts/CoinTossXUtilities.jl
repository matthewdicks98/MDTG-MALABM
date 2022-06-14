#=
CoinTossXUtilities:
- Julia version: 1.5.3
- Authors: Matthew Dicks, Ivan Jericevich, Patrick Chang, Tim Gebbie
- Function: Provide the necessary functions for running simulations with CoinTossX
- Structure:
    1. Build, deploy, start CoinTossX and initialise Java Virtual Machine with required byte code paths
    2. Order creation structure
    3. Client structure
    4. Initialize client by logging in to the trading gateway and starting the trading session
	5. Reset LOB
    6. Submit an order to CoinTossX
    7. Cancel an existing order
    8. Receive updates to the best bid/ask
    9. Shutdown all components of CoinTossX
=#
ENV["JULIA_COPY_STACKS"]=1
using JavaCall
ctx_directory = "/home/matt/IdeaProjects"
software_path = "/home/matt/IdeaProjects/CoinTossXIvan_Run"
#---------------------------------------------------------------------------------------------------

#----- Build, deploy, start CoinTossX and initialise Java Virtual Machine with required byte code paths -----#
function StartCoinTossX(; build::Bool = true, deploy::Bool = true)

    # build and deploy
    cd(ctx_directory * "/CoinTossXIvan")
	if build
        println("--------------------------------Building CoinTossX--------------------------------")
		run(`./gradlew -Penv=local build -x test`)
	end
    if deploy
        println("--------------------------------Deploying CoinTossX--------------------------------")
		run(`./gradlew -Penv=local clean installDist bootWar copyResourcesToInstallDir copyToDeploy deployLocal`)
	end

    # start the application
    cd(software_path * "/scripts")
    println("--------------------------------Starting CoinTossX--------------------------------")
    run(`./startAll.sh`)
	println("--------------------------------CoinTossX has started--------------------------------")
    cd(ctx_directory)
end

function StopCoinTossX()
    cd(software_path * "/scripts")
	try
		run(`./stopAll.sh`)
	catch e
		println(e)
	end
    println("--------------------------------CoinTossX has stopped--------------------------------")
end

function Deploy()
    # can be used to redeploy CTX which will free up memory after the simulations are done
    cd(ctx_directory * "/CoinTossXIvan")
    println("--------------------------------Deploying CoinTossX--------------------------------")
	run(`./gradlew -Penv=local clean installDist bootWar copyResourcesToInstallDir copyToDeploy deployLocal`)
end

function StartJVM()
    JavaCall.addClassPath(ctx_directory * "/CoinTossXIvan/ClientSimulator/build/classes/main")
    JavaCall.addClassPath(ctx_directory * "/CoinTossXIvan/ClientSimulator/build/install/ClientSimulator/lib/*.jar")
    # JavaCall.init()
    # "-XX:+UseLargePages",
    # "-server",
    JavaCall.init(["-Xmx2G", "-Xms2G", "-d64", "-XX:+UseStringDeduplication", "-Dagrona.disable.bounds.checks=true", "-XX:+UseG1GC", "-XX:+OptimizeStringConcat", "-XX:+UseCondCardMark"])
    println("--------------------------------JVM has started--------------------------------")
end
#---------------------------------------------------------------------------------------------------

#----- CoinTossX Client structure -----#
mutable struct TradingGateway
    id::Int64
    securityId::Int64
    javaObject::JavaObject{Symbol("client.Client")}
end
#---------------------------------------------------------------------------------------------------

#----- Initialize client by logging in to the trading gateway and starting the trading session. Also logout. -----#
function Login(clientId::Int64, securityId::Int64)
    utilities = @jimport example.Utilities
    javaObject = jcall(utilities, "loadClientData", JavaObject{Symbol("client.Client")}, (jint, jint), clientId, securityId)
    println("--------------------------------Logged in and trading session started--------------------------------")
    return TradingGateway(clientId, securityId, javaObject)
end

function Login!(tradingGateway::TradingGateway)
    utilities = @jimport example.Utilities
    tradingGateway.javaObject = jcall(utilities, "loadClientData", JavaObject{Symbol("client.Client")}, (jint, jint), tradingGateway.id, tradingGateway.securityId)
    println("--------------------------------Logged in and trading session started--------------------------------")
end

function Logout(tradingGateway::TradingGateway)
    jcall(tradingGateway.javaObject, "close", Nothing, ())
    println("--------------------------------Logged out and trading session ended--------------------------------")
end

#---------------------------------------------------------------------------------------------------

#----- Set and Reset LOB -----#
function StartLOB(tradingGateway::TradingGateway)
	jcall(tradingGateway.javaObject, "sendStartMessage", Nothing, ())
end
function EndLOB(tradingGateway::TradingGateway)
	jcall(tradingGateway.javaObject, "sendEndMessage", Nothing, ())
end
#---------------------------------------------------------------------------------------------------

#----- Order creation structure -----#
mutable struct Order
    orderId::Int64
    traderMnemonic::String
    volume::Int64
    price::Int64
    side::String
    type::String
    tif::String
    displayVolume::Int64
    mes::Int64
    stopPrice::Int64
    expireTime::String
    function Order(; orderId::Int64 = 0, traderMnemonic::String = "", side::String = "", type::String = "", volume::Int64 = 0, price::Int64 = 0, tif::String = "Day", displayVolume = missing, mes::Int64 = 0, stopPrice::Int64 = 0, expireTime::String = "20221230-23:00:00")
        if ismissing(displayVolume)
            new(orderId, traderMnemonic, volume, price, side, type, tif, volume, mes, stopPrice, expireTime)
        else
            new(orderId, traderMnemonic, volume, price, side, type, tif, displayVolume, mes, stopPrice, expireTime)
        end
    end
    Order() = new(0, "", 0, 0, "", "", "", 0, 0, 0, "")
end
#---------------------------------------------------------------------------------------------------

#----- Submit an order to CoinTossX -----#
function SubmitOrder(tradingGateway::TradingGateway, order::Order)
    jcall(tradingGateway.javaObject, "submitOrder", Nothing, (jint, JString, jlong, jlong, JString, JString, JString, JString, jlong, jlong, jlong), Int32(order.orderId), order.traderMnemonic, order.volume, order.price, order.side, order.type, order.tif, order.expireTime, order.displayVolume, order.mes, order.stopPrice)
end
#---------------------------------------------------------------------------------------------------

#----- Cancel an existing order -----#
function CancelOrder(tradingGateway::TradingGateway, order::Order)
    jcall(tradingGateway.javaObject, "cancelOrder", Nothing, (jint, JString, JString, jlong), Int32(order.orderId), order.traderMnemonic, order.side, order.price)
end
#---------------------------------------------------------------------------------------------------

