import React, { PropTypes } from 'react';
import ReactDom from 'react-dom';
import { connect } from 'react-redux'

class NetStatistics extends React.Component {

  constructor(props) {
    super(props);
    this.createNetStatisticsChart = this.createNetStatisticsChart.bind(this);
    // this.state = DEFAULT_STATE;
  }

  componentDidMount(){
  };

  onClearButtonClick(){
  };

  createNetStatisticsChart() {
    const netStatistics = this.props.netStatistics;
    console.log("nodeValues: ", netStatistics)
    return(
      netStatistics.map( (netStat, i) => {
        return <div key={i}> { val }</div>
      })
    )
  }

  render() {
    const name = this.props.name;
    const prediction = this.props.prediction;

    return (
      <div className="net-statistics">
          {
            this.props.netStatistics.map( (netStat, i) => {
              return <div key={"netStatList" + i}>
                <ul>
                  <li> name: { netStat.name } </li>
                  <li> prediction: { netStat.prediciton } </li>
                </ul>
              </div>
            })
          }
      </div>
    );
  }
};

const mapStateToProps = (state) => {
  return {
    netStatistics: state.netStatistics
  }
};

NetStatistics.defaultProps = { };

const mapDispatchToProps = (dispatch) => {
  return {
    // dispatchAddTodo(todo) {
    //   dispatch(postTodoToDB(todo))
    // }
  }
};

export default connect(mapStateToProps, mapDispatchToProps)(NetStatistics);
