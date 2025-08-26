#! /usr/bin/env ruby
# coding: utf-8
## -*- mode: ruby -*-
## = RubyAgentBase for CrowdWalk
## Author:: Itsuki Noda
## Version:: 0.0 2015/06/27 I.Noda
##
## === History
## * [2015/06/27]: Create This File.
## * [YYYY/MM/DD]: add more
## == Usage
## * ...

require 'ItkUtility.rb' ;
require 'ItkTerm.rb' ;
require 'NetworkMap.rb' ;

#--======================================================================
#++
## CrowdWalk の RubyAgent に対応する Ruby 側の AgentBase
class RubyAgentBase
  include ItkUtility ;

  #--============================================================
  #--::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  #++
  ## Java から Ruby を呼び出すTriggerでのFilter。
  ## この配列に Java のメソッド名（キーワード）が入っていると、
  ## Ruby 側が呼び出される。入っていないと、無視される。
  ## RubyAgentBase を継承するクラスは、このFilterを持つことが望ましい。
  ## このFilterは、クラスをさかのぼってチェックされる。
  TriggerFilter = [
#                   "preUpdate",
#                   "update",
#                   "calcCostFromNodeViaLink",
#                   "calcSpeed",
#                   "calcAccel",
#                   "thinkCycle",
                  ] ;

  #--@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  #++
  ## Java 側の Agent オブジェクト
  attr_accessor :javaAgent ;

  ## config 情報。generation ファイルでの設定がそのまま渡る。
  attr_accessor :config ;

  ## fallback 情報。
  attr_accessor :fallback ;

  #--------------------------------------------------------------
  #++
  ## 初期化
  ## _agent_:: Java の RubyAgent のインスタンス。
  ## _initOption_:: 初期化のためのオプション引数。
  ##                generation file で指定できる。指定しなければ nil。
  def initialize(agent, config, fallback)
    @javaAgent = agent ;
    @config = config ;
    @fallback = fallback ;
  end

  #--------------------------------------------------------------
  # 変数アクセス関係
  #--------------------------------------------------------------
  #++
  ## エージェント id
  def getAgentId()
    return @javaAgent.getID() ;
  end

  #--------------------------------------------------------------
  #++
  ## 現在時刻
  def getCurrentTime()
    return @javaAgent.currentTime ;
  end

  #--------------------------------------------------------------
  #++
  ## 現在地のリンク
  def getCurrentLink()
    return @javaAgent.getCurrentLink()
  end

  #--------------------------------------------------------------
  #++
  ## 現在地のリンク ID
  def getCurrentLinkId()
    return getCurrentLink().ID ;
  end

  #--------------------------------------------------------------
  #++
  ## 速度
  def getCurrentSpeed()
    return @javaAgent.getSpeed() ;
  end

  #--------------------------------------------------------------
  #++
  ## 自由速度
  def getEmptySpeed()
    return @javaAgent.getEmptySpeed() ;
  end

  #--------------------------------------------------------------
  #++
  ## 自由速度を設定
  ## _speed_ :: 新しい自由速度。double に変換可能であること。
  def setEmptySpeed(speed)
    return @javaAgent.setEmptySpeed(speed.to_f) ;
  end

  #--------------------------------------------------------------
  # 目的地および経路関係
  #--------------------------------------------------------------
  #++
  ## 目的地
  def getGoal()
    return ItkTerm.toRuby(@javaAgent.getGoal()) ;
  end

  #--------------------------------------------------------------
  #++
  ## 目的地設定
  ## _goalTag_ :: 新しいゴールのタグ（String）
  def setGoal(goalTag)
    @javaAgent.changeGoal(ItkTerm.ensureTerm(goalTag)) ;
  end

  #--------------------------------------------------------------
  #++
  ## ルート取得
  ## _future_ :: 今後のルートのみの場合。
  def getRoute(future = true)
    routePlan = @javaAgent.routePlan ;
    baseRoute = routePlan.getRoute() ;
    beginIndex = (fugure ? routePlan.getIndex() : 0) ;
    endIndex = baseRoute.size() ;

    route = [] ;
    (beginIndex ... endIndex).each{|index|
      route.push(ItkTerm.toRuby(baseRoute[index])) ;
    }
    return route ;
  end

  #--------------------------------------------------------------
  #++
  ## ルートへの挿入。
  ## _newRoute_ :: 挿入する今後のルート。中継点のタグもしくはタグリスト。
  ## _clearP_ :: 挿入前にクリアするかどうか。
  ##             true なら、setRoute のような動き。
  def insertRoute(newRoute, clearP = false)
    @javaAgent.clearPlannedRoute() if(clearP) ;

    if(newRoute.is_a?(Array)) then
      newRoute.each{|subgoal|
        insertRoute(subgoal, false) ;
      }
    else
      @javaAgent.insertRouteTagSafelyForRuby(newRoute) ;
    end
  end

  #--------------------------------------------------------------
  #++
  ## ルートのクリア
  def clearRoute()
    insertRoute([], true) ;
  end

  #--------------------------------------------------------------
  #++
  ## 方向変換(turn around)
  ## 方向変換を予定する。
  ## 次の preUpdate() で反映される。
  ## 2回やっても、元には戻らない。（turn around がセットされるだけなので）
  def setTurnAround()
    @javaAgent.setTurnAround() ;
  end

  #--------------------------------------------------------------
  # タグ関係
  #--------------------------------------------------------------
  #++
  ## エージェントのタグリスト
  def getAgentTags()
    return @javaAgent.getTags() ;
  end

  #--------------------------------------------------------------
  #++
  ## エージェントのタグを持つかどうか
  def hasAgentTag(tag)
    return @javaAgent.hasTag(tag) ;
  end

  #--------------------------------------------------------------
  #++
  ## エージェントのタグを追加
  def addAgentTag(tag)
    return @javaAgent.addTag(tag) ;
  end

  #--------------------------------------------------------------
  #++
  ## エージェントのタグを削除
  def removeAgentTag(tag)
    return @javaAgent.removeTag(tag) ;
  end

  #--------------------------------------------------------------
  #++
  ## 現在地のリンクのタグリスト
  def getPlaceTags()
    return getCurrentLink().getTags() ;
  end

  #--------------------------------------------------------------
  #++
  ## 現在地のリンクのタグをチェック
  def hasPlaceTag(tag)
    # return getCurrentLink().hasTag(tag) ;
    # for speed up
    return @javaAgent.hasPlaceTagForRuby(tag) ;
  end

  #--------------------------------------------------------------
  #++
  ## 現在地のリンクにタグを追加
  def addPlaceTag(tag)
    return getCurrentLink().addTag(tag) ;
  end

  #--------------------------------------------------------------
  #++
  ## アラートテーブル
  def getAlertTable()
    return @javaAgent.alertedMessageTable ;
  end

  #--------------------------------------------------------------
  #++
  ## アラートを聴いた時刻。
  ## 聴いていなければ、nilが返る。
  ## _message_ :: アラートメッセージ。文字列もしくは Itk.Term.
  ## *return* 時刻もしくは nil
  def listenAlert(message)
#    return getAlertTable().get(ItkTerm.ensureTerm(message)) ;
    ## 高速化のためのショートカット
    return @javaAgent.getAlertTimeForRuby(message) ;
  end

  #--------------------------------------------------------------
  #++
  ## アラートを記憶。
  def saveAlert(message, redundant = false)
    messageTerm = ItkTerm.ensureTerm(message);
    pastTime = getAlertTable().get(messageTerm) ;

    if(redundant || pastTime.nil?) then
      getAlertTable().put(messageTerm, getCurrentTime()) ;
      return true ;
    else
      return false ;
    end
  end

  #--------------------------------------------------------------
  #++
  ## アラートを現在地にアナウンス。
  def announceAlert(message)
    messageTerm = ItkTerm.ensureTerm(message);
    getCurrentLink.addAlertMessage(messageTerm, getCurrentTime(), true) ;
    return true ;
  end

  #--------------------------------------------------------------
  #++
  ## 現在Agent自身が保持している message を消去する。
  def clearAlert(message)
    getAlertTable().remove(message) ;
  end

  #--------------------------------------------------------------
  #++
  ## 現在Agent自身が保持している全 message を消去する。
  def clearAllAlert()
    getAlertTable().clear() ;
  end

  #--------------------------------------------------------------
  # トリガされたメソッド
  #--============================================================
  #--------------------------------------------------------------
  #++
  ## RubyAgentBase を継承するクラスにおいて、TriggerFilter に methodName
  ## 含まれているかをチェックする。
  ## このクラスメソッドが true を返すもののみ、
  ## java から ruby のメソッドが呼ばれる。
  ## java と ruby の行き来のオーバーヘッドを軽くするための措置。
  def self.checkTriggerFilter(methodName)
    # TriggerFilter が定義されていて、methodName を含めば、true を返す
    if(self.const_defined?(:TriggerFilter)) then
      if(self::TriggerFilter.include?(methodName)) then
        return true ;
      end
    end
    # RubyAgentBase までさかのぼっていれば、探索撃ち切って false を返す。
    if(self == RubyAgentBase) then
      return false ;
    end
    # それ以外は、親クラスを探しに行く。
    return self.superclass().checkTriggerFilter(methodName) ;
  end

  #--------------------------------------------------------------
  #++
  ## シミュレーション各サイクルの前半に呼ばれる。
  def preUpdate()
    return @javaAgent.super_preUpdate() ;
  end

  #--------------------------------------------------------------
  #++
  ## シミュレーション各サイクルの後半に呼ばれる。
  ## _simTime_:: 相対シミュレーション時刻
  def update()
    return @javaAgent.super_update() ;
  end

  #--------------------------------------------------------------
  #++
  ## あるwayを選択した場合の目的地(_target)までのコスト。
  ## _way_:: 現在進もうとしている道
  ## _node_:: 現在の分岐点
  ## _target_:: 最終目的地
  def calcCostFromNodeViaLink(way, node, target)
#    p [:calcCostFromNodeViaLink, getAgentId(), way, node, target] ;
    return @javaAgent.super_calcCostFromNodeViaLink(way, node, target);
  end

  #--------------------------------------------------------------
  #++
  ## 速度を計算する。
  ## _previousSpeed_:: 前のサイクルの速度。
  ## *return* 速度。
  def calcSpeed(previousSpeed)
#    p [:calcSpeed, getAgentId(), previousSpeed] ;
    return @javaAgent.super_calcSpeed(previousSpeed) ;
  end

  #--------------------------------------------------------------
  #++
  ## 加速度を計算する。
  ## _baseSpeed_:: 自由速度。
  ## _previousSpeed_:: 前のサイクルの速度。
  ## *return* 加速度。
  def calcAccel(baseSpeed, previousSpeed)
#    p [:calcAccel, getAgentId(), baseSpeed, previousSpeed] ;
    return @javaAgent.super_calcAccel(baseSpeed, previousSpeed) ;
  end

  #--------------------------------------------------------------
  #++
  ## 思考ルーチン
  def thinkCycle()
    return @javaAgent.super_thinkCycle() ;
  end

  #--------------------------------------------------------------
  #++
  ## 文字列の java 内でのintern.
  def intern(str)
    return @javaAgent.intern(str) ;
  end

  #--------------------------------------------------------------
  #++
  ## Itkのloggerによるログ出力.
  ## ItkUtility のものを override.
  ## _level_ :: ログレベル。:trace, :debug, :info, :warn, :error, :fatal
  ## _label_ :: ログのラベル。nil なら、Agent ID などに置き換えられる。
  ## _*data_ :: データの並び。
  def logWithLevel(level, label, *data)
    levelObj = ItkUtility::LogLevelTable[level] ;
    return @javaAgent.logAsRubyAgent(levelObj, label, *data) ;
  end

  #--============================================================
  #--::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  #--@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  #--------------------------------------------------------------

end # class RubyAgentBase
