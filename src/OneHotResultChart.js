import React, { PropTypes } from 'react';
import ReactDom from 'react-dom';
import { connect } from 'react-redux'
import ReactHighcharts from 'react-highcharts';

class OneHotResultChart extends React.Component {

  constructor(props) {
    super(props);
    console.log("props: ", props)
    this.createOneHotResultChartChart = this.createOneHotResultChartChart.bind(this);
    this.formatChartData = this.formatChartData.bind(this);
  }

  componentDidMount() {
    let chart = this.refs.chart.getChart();
  }

  onClearButtonClick(){
  };

  formatChartData(name, oneHotResult) {
    return [{
      name: name,
      data: oneHotResult,
    }]
  }

  createOneHotResultChartChart() {
    const netStatistics = this.props.netStatistics;
    console.log("nodeValues: ", netStatistics)
    return(
      netStatistics.map( (netStat, i) => {
        return <div key={i}> { val }</div>
      })
    )
  }

  render() {
    const config = {
      chart: {
          type: 'column'
      },
      title: {
        text: 'Model: ' + this.props.name,
      },
      xAxis: {
        categories: [
          '0',
          '1',
          '2',
          '3',
          '4',
          '5',
          '6',
          '7',
          '8',
          '9',
        ],
        crosshair: true
      },
      yAxis: {
        title: {
            text: 'Node Strength'
        }
      },
      plotOptions: {
        column: {
          pointPadding: 0.2,
          borderWidth: 0
        }
      },
      series: this.formatChartData(this.props.name, this.props.oneHotResult)
    }

    return (
      <div className="one-hot-result-container">
        { <ReactHighcharts config={config} ref="chart" /> }
      </div>
    );
  }
};

const mapStateToProps = (state) => {
  return {
    netStatistics: state.netStatistics
  }
};

OneHotResultChart.defaultProps = { };

const mapDispatchToProps = (dispatch) => {
  return {
    // dispatchAddTodo(todo) {
    //   dispatch(postTodoToDB(todo))
    // }
  }
};

export default connect(mapStateToProps, mapDispatchToProps)(OneHotResultChart);
